import copy
import math
import warnings

import torch

import pytorch3d.ops
from pytorch3d.loss.point_mesh_distance import point_face_distance
import point_cloud_utils as pcu

from src.mesh.TMesh import get_transformation_matrix_from_rot_and_trans
from src.model import PointDistributionModel, BatchedPointDistributionModel
from src.registration import ProcrustesAnalyser, ICPAnalyser
from src.sampling.proposals import ParameterProposalType
from src.sampling.proposals.ClosestPoint import extract_bounds_verts, is_point_on_line_segment


def unnormalised_log_posterior(distances, inv_cov_operator, parameters, translation, rotation, gamma, sigma_lm,
                               sigma_mod,
                               uniform_pose_prior=True, sigma_trans=20.0, sigma_rot=0.005):
    """
    Calculates the unnormalised log posterior given the distance of each point of the target to its closest point on the
    surface of the current shape and the model parameters of the current shape. It is possible to use a regularisation
    matrix to regularise the distance vector.
    This method can also be called for entire batches of shapes.
    The prior term pushes the solution towards a more likely shape by penalizing unlikely shape deformations. A
    zero-mean Gaussian distribution with variances corresponding to the input parameters is assumed for all 3 types of
    parameters. For the likelihood term, the L2 distance (independent point evaluator likelihood) is used. The L2
    distance is also evaluated using a zero-mean Gaussian with indicated variance.

    :param distances: Tensor with distances from each point of the target to the closest point on the surface of the
    current shapes considered with shape (num_target_points, batch_size).
    :type distances: torch.Tensor
    :param inv_cov_operator: Inverse of a covariance operator to regularise the distance vector with shape
    (num_target_points, num_target_points), see the paper F. Steinke and B. Schölkopf - Kernels, regularization and
    differential equations for more in-depth information.
    :type inv_cov_operator: torch.Tensor
    :param parameters: Tensor with the current model parameters of the current shapes considered with shape
    (num_parameters, batch_size).
    :type parameters: torch.Tensor
    :param translation: Tensor with the current translation parameters of the current shapes considered with shape
    (3, batch_size).
    :type translation: torch.Tensor
    :param rotation: Tensor with the current rotation parameters of the current shapes considered with shape
    (3, batch_size).
    :type rotation: torch.Tensor
    :param gamma: Multiplication factor of the likelihood term.
    :type gamma: float
    :param sigma_lm: Variance of the zero-mean Gaussian, which is used to evaluate the L2 distances. The same variance
    value is used for all 3 dimensions (isotropic distribution).
    :type sigma_lm: float
    :param sigma_mod: Variance of the model parameters (prior term calculation).
    :type sigma_mod: float
    :param uniform_pose_prior: If True, an uninformed prior distribution is used for the pose parameters. If False, then
    a zero-mean Gaussian distribution with variances var_trans, var_rot is assumed.
    :type uniform_pose_prior: bool
    :param sigma_trans: Variance of the translation parameters (prior term calculation).
    :type sigma_trans: float
    :param sigma_rot: Variance of the rotation parameters (prior term calculation).
    :type sigma_rot: float
    :return: Tensor with the unnormalised log posterior values of the analysed shapes with shape (batch_size,).
    :rtype: torch.Tensor
    """
    residual = torch.mul(distances, inv_cov_operator @ distances)
    # sigma_lm = (torch.linalg.vector_norm(inv_cov_operator @ distances) / torch.linalg.vector_norm(distances)) * sigma_lm
    log_likelihoods = 0.5 * gamma * (torch.mean(-residual, dim=0) / sigma_lm)
    log_prior = 0.5 * torch.sum(-torch.pow(parameters / sigma_mod, 2), dim=0)
    rotation = (rotation + math.pi) % (2.0 * math.pi) - math.pi
    if uniform_pose_prior:
        log_translation = torch.zeros_like(log_likelihoods)
        log_rotation = torch.zeros_like(log_likelihoods)
    else:
        log_translation = 0.5 * torch.sum(-torch.pow(translation / sigma_trans, 2), dim=0)
        log_rotation = 0.5 * torch.sum(-torch.pow(rotation / sigma_rot, 2), dim=0)
    return log_likelihoods + log_prior + log_translation + log_rotation


def unnormalised_log_posterior_curvature(distances, inv_cov_operator, parameters, translation, rotation, gamma,
                                         sigma_lm, sigma_mod, rel_curvs, uniform_pose_prior=True, sigma_trans=20.0,
                                         sigma_rot=0.005):
    """
    Same method as above, but the individual independent point contributions are weighted with the local curvature when 
    calculating the likelihood term.
    
    :param distances: Tensor with distances from each point of the target to the closest point on the surface of the
    current shapes considered with shape (num_target_points, batch_size).
    :type distances: torch.Tensor
    :param inv_cov_operator: Inverse of a covariance operator to regularise the distance vector with shape
    (num_target_points, num_target_points), see the paper F. Steinke and B. Schölkopf - Kernels, regularization and
    differential equations for more in-depth information.
    :type inv_cov_operator: torch.Tensor
    :param parameters: Tensor with the current model parameters of the current shapes considered with shape
    (num_parameters, batch_size).
    :type parameters: torch.Tensor
    :param translation: Tensor with the current translation parameters of the current shapes considered with shape
    (3, batch_size).
    :type translation: torch.Tensor
    :param rotation: Tensor with the current rotation parameters of the current shapes considered with shape
    (3, batch_size).
    :type rotation: torch.Tensor
    :param gamma: Multiplication factor of the likelihood term.
    :type gamma: float
    :param sigma_lm: Variance of the zero-mean Gaussian, which is used to evaluate the L2 distances. The same variance
    value is used for all 3 dimensions (isotropic distribution).
    :type sigma_lm: float
    :param sigma_mod: Variance of the model parameters (prior term calculation).
    :type sigma_mod: float
    :param rel_curvs: Measure for the relative curvature at each point with shape (num_target_points, batch_size).
    :type rel_curvs: torch.Tensor
    :param uniform_pose_prior: If True, an uninformed prior distribution is used for the pose parameters. If False, then
    a zero-mean Gaussian distribution with variances var_trans, var_rot is assumed.
    :type uniform_pose_prior: bool
    :param sigma_trans: Variance of the translation parameters (prior term calculation).
    :type sigma_trans: float
    :param sigma_rot: Variance of the rotation parameters (prior term calculation).
    :type sigma_rot: float
    :return: Tensor with the unnormalised log posterior values of the analysed shapes with shape (batch_size,).
    :rtype: torch.Tensor
    """
    residual = torch.mul(distances, inv_cov_operator @ distances)
    # sigma_lm = (torch.linalg.vector_norm(inv_cov_operator @ distances) / torch.linalg.vector_norm(distances)) * sigma_lm
    log_likelihoods = 0.5 * gamma * (torch.sum(torch.mul((-residual / sigma_lm), rel_curvs), dim=0))
    log_prior = 0.5 * torch.sum(-torch.pow(parameters / sigma_mod, 2), dim=0)
    rotation = (rotation + math.pi) % (2.0 * math.pi) - math.pi
    if uniform_pose_prior:
        log_translation = torch.zeros_like(log_likelihoods)
        log_rotation = torch.zeros_like(log_likelihoods)
    else:
        log_translation = 0.5 * torch.sum(-torch.pow(translation / sigma_trans, 2), dim=0)
        log_rotation = 0.5 * torch.sum(-torch.pow(rotation / sigma_rot, 2), dim=0)
    return log_likelihoods + log_prior + log_translation + log_rotation


def get_laplacian(type, verts, faces, edges, alpha, beta, dev):
    size = verts.size(0)
    identity = torch.diag(torch.ones(size, device=dev))
    if type == "none":
        return identity, torch.ones(size, device=dev), - (size / 2) * torch.log(torch.tensor(2 * torch.pi, device=dev))
    if type == "std" or type == "base" or type == "uniform":
        graph_laplacian = pytorch3d.ops.laplacian(verts, edges).to_dense()
        degree = (graph_laplacian.abs() > 0.0).sum(-1) - 1
        degree = torch.clamp(degree, min=1)
        graph_laplacian *= degree[:, None]
        graph_laplacian = -torch.pow(degree, -0.5)[None, :] * graph_laplacian * torch.pow(degree, -0.5)[:, None]
    if type == "cot":
        graph_laplacian = -pytorch3d.ops.cot_laplacian(verts, faces)[0].to_dense()
        sums = -graph_laplacian.sum(dim=1)
        D = torch.diag(torch.where(sums > 0, 1.0 / torch.sqrt(sums), torch.zeros_like(sums)))
        graph_laplacian[torch.arange(graph_laplacian.size(0)), torch.arange(graph_laplacian.size(0))] = sums
        graph_laplacian = D @ graph_laplacian @ D
    if type == "norm":
        graph_laplacian = -pytorch3d.ops.norm_laplacian(verts, edges).to_dense()
        sums = -graph_laplacian.sum(dim=1)
        D = torch.diag(torch.where(sums > 0, 1.0 / torch.sqrt(sums), torch.zeros_like(sums)))
        graph_laplacian[torch.arange(graph_laplacian.size(0)), torch.arange(graph_laplacian.size(0))] = sums
        graph_laplacian = D @ graph_laplacian @ D
    if type == "eps":
        dists, idx, knn = pytorch3d.ops.ball_query(verts.unsqueeze(-1).permute(2, 0, 1),
                                                   verts.unsqueeze(-1).permute(2, 0, 1), K=size, radius=30.0)
        dists, idx = dists[0, :, :], idx[0, :, :]
        rows = torch.arange(dists.size(0), device=dev).unsqueeze(1).expand_as(dists)
        idx[idx == -1] = rows[idx == -1]
        graph_laplacian = torch.zeros((size, size), device=dev).scatter_(1, idx, dists)
        graph_laplacian = torch.where(graph_laplacian > 0, -torch.exp(-graph_laplacian / 900.0), graph_laplacian)
        graph_laplacian = graph_laplacian / torch.norm(graph_laplacian, p=2, dim=1, keepdim=True)
        graph_laplacian[torch.arange(graph_laplacian.size(0)), torch.arange(graph_laplacian.size(0))] = 1.0

    prec = torch.matrix_power((0.5 * identity + beta * graph_laplacian), alpha)
    eigvals = (1.0 / (torch.pow(0.5 + beta * torch.linalg.eigh(graph_laplacian)[0], alpha) + 1e-6))
    # prec *= eigvals.mean()
    normalization = -0.5 * torch.log(eigvals / eigvals.mean() + 1e-6).sum(-1) - (size / 2) * torch.log(
        torch.tensor(2 * torch.pi, device=dev))
    return prec, eigvals, normalization


class PDMMetropolisSampler:
    def __init__(self, pdm, proposal, batch_mesh, target, laplacian_type, alpha=2, beta=0.3, fixed_correspondences=True,
                 triangles=None, barycentric_coords=None, gamma=50.0, var_like=1.0, var_prior_mod=1.0,
                 uniform_pose_prior=True, var_prior_trans=20.0, var_prior_rot=0.005, save_full_mesh_chain=False,
                 save_residuals=False):
        """
        Main class for Bayesian model fitting using Markov Chain Monte Carlo (MCMC). The Metropolis algorithm
        allows the user to draw samples from any distribution, given that the unnormalized distribution can be evaluated
        point-wise. This requirement is easy to fulfill for the shape modelling applications studied in the context of
        this application.

        :param pdm: The point distribution model (PDM) to be fitted to a given target.
        :type pdm: PointDistributionModel
        :param proposal: Proposal instance that defines how the new parameters are drawn for a new proposal.
        :type proposal: GaussianRandomWalkProposal
        :param batch_mesh: Batch of meshes, which is used to construct the instances defined by the parameters and
        compare them with the target (i.e., calculating the unnormalised posterior value).
        :type batch_mesh: BatchTorchMesh
        :param target: Observed (partial) shape to be analysed. Remark: It is required that the target has the same
        ‘batch_size’ as ‘batch_mesh’.
        :type target: BatchTorchMesh
        :param fixed_correspondences: Boolean variable that determines whether there are known point fixed_correspondences between
        the model and the target.
        :type fixed_correspondences: bool
        :param gamma: Multiplication factor of the likelihood term.
        :type gamma: float
        :param var_like: Variance used (in square millimetres) when determining the likelihood term of the posterior.
        For the likelihood term, the L2 distance (independent point evaluator likelihood) is used. The L2 distance is
        evaluated using a zero-mean Gaussian with variance var_like.
        :type var_like: float
        :param var_prior_mod: Variance used when determining the prior term of the posterior. The prior term pushes the
        solution towards a more likely shape by penalizing unlikely shape deformations.
        :type var_prior_mod: float
        :param uniform_pose_prior: If True, an uninformed prior distribution is used for the pose parameters when
        determining the prior term of the posterior. If False, a zero-mean Gaussian distribution with variances
        var_trans, var_rot is assumed.
        :type uniform_pose_prior: bool
        :param var_prior_trans: Variance of the translation parameters (prior term calculation).
        :type var_prior_trans: float
        :param var_prior_rot: Variance of the rotation parameters (prior term calculation).
        :type var_prior_rot: float
        :param save_full_mesh_chain: If True, the posterior meshes are saved. This requires a lot of memory.
        :type save_full_mesh_chain: bool
        :param save_residuals: If True, then the residuals are saved.
        :type save_residuals: bool
        """
        self.model = pdm
        self.proposal = proposal
        # Make sure all components are on the same device, i.e., model, proposal, meshes.
        self.dev = self.proposal.dev
        self.batch_mesh = batch_mesh
        self.points = self.batch_mesh.tensor_points
        self.target = target
        self.target_points = self.target.tensor_points
        self.fixed_correspondences = fixed_correspondences
        self.triangles = triangles
        self.barycentric_coords = barycentric_coords
        self.batch_size = self.proposal.batch_size
        self.gamma = gamma
        self.var_like = var_like
        self.var_prior_mod = var_prior_mod
        self.uniform_pose_prior = uniform_pose_prior
        self.var_prior_trans = var_prior_trans
        self.var_prior_rot = var_prior_rot
        if self.uniform_pose_prior:
            self.var_prior_trans, self.var_prior_rot = 0.0, 0.0

        self.target_clouds = self.target.to_pytorch3d_pointclouds()
        self.full_target_clouds = None

        # TODO: Introduce input variables to control curvature-dependent weighting from the main script.
        # Mean curvature is the average of the 2 principal curvatures.
        # curvatures, directions = pytorch3d.ops.estimate_pointcloud_local_coord_frames(
        #    self.target_points.permute(2, 0, 1))
        # mean_curv = curvatures[:, :, 1:3].mean(dim=2) / curvatures[:, :, 1:3].mean(dim=2).sum(dim=1, keepdim=True)
        # Use the reciprocals of the relative curvatures as weights
        # self.mean_curv = (1 / (mean_curv.permute(1, 0) + 1e-8)) / (1 / (mean_curv.permute(1, 0) + 1e-8)).sum(dim=0, keepdim=True)
        # self.mean_curv = mean_curv.permute(1, 0)

        # PyTorch3D function leaves all diagonal elements at 0.
        # dists, idx, knn = pytorch3d.ops.ball_query(self.target_points[:, :, 0].unsqueeze(-1).permute(2, 0, 1),
        #                                           self.target_points[:, :, 0].unsqueeze(-1).permute(2, 0, 1),
        #                                           K=self.target.num_points, radius=30.0)
        # dists, idx = dists[0, :, :], idx[0, :, :]
        # rows = torch.arange(dists.size(0), device=self.dev).unsqueeze(1).expand_as(dists)
        # idx[idx == -1] = rows[idx == -1]
        # graph_laplacian = torch.zeros((self.target.num_points, self.target.num_points), device=self.dev).scatter_(1, idx, dists)
        # graph_laplacian = torch.where(graph_laplacian > 0, -torch.exp(-graph_laplacian / 900.0), graph_laplacian)

        target_meshes = self.target.to_pytorch3d_meshes()
        edges = target_meshes.edges_packed()[
                target_meshes.mesh_to_edges_packed_first_idx()[0]:target_meshes.mesh_to_edges_packed_first_idx()[1], :]
        self.laplacian_type = laplacian_type
        self.alpha = alpha
        self.beta = beta
        self.inv_cov_operator, eigvals, det = get_laplacian(self.laplacian_type, self.target_points[:, :, 0],
                                                            self.target.cells[0].data, edges, self.alpha, self.beta,
                                                            self.dev)

        self.posterior = None
        self.determine_quality(ParameterProposalType.MODEL_RANDOM)
        self.old_posterior = self.posterior
        self.accepted_par = torch.zeros(self.batch_size, device=self.dev)
        self.accepted_rnd = torch.zeros(self.batch_size, device=self.dev)
        self.accepted_trans = torch.zeros(self.batch_size, device=self.dev)
        self.accepted_rot = torch.zeros(self.batch_size, device=self.dev)
        self.rejected_par = torch.zeros(self.batch_size, device=self.dev)
        self.rejected_rnd = torch.zeros(self.batch_size, device=self.dev)
        self.rejected_trans = torch.zeros(self.batch_size, device=self.dev)
        self.rejected_rot = torch.zeros(self.batch_size, device=self.dev)
        self.save_chain = save_full_mesh_chain
        self.full_chain = []
        self.save_residuals = save_residuals
        self.residuals_c = []
        self.residuals_n = []

    def propose(self, parameter_proposal_type: ParameterProposalType):
        """
        Implements the first step of the Metropolis algorithm (according to the lecture notes of the course "Statistical
        Shape Modelling" by Marcel Lüthi). Draws new samples from the proposal distribution, defined by the proposal
        instance.

        :param parameter_proposal_type: Specifies which parameters (model, translation or rotation) are to be drawn.
        :type parameter_proposal_type: ParameterProposalType
        """
        self.proposal.propose(parameter_proposal_type)

    def update_mesh(self, parameter_proposal_type: ParameterProposalType):
        """
        Updates the meshes according to the new parameter values previously drawn. The old mesh points are saved so that
        they can be reset if the new sample is rejected.

        :param parameter_proposal_type: Specifies which parameters (model, translation or rotation) were drawn during
        the current iteration.
        :type parameter_proposal_type: ParameterProposalType
        """
        old_points = self.batch_mesh.tensor_points
        if parameter_proposal_type == ParameterProposalType.MODEL_RANDOM or parameter_proposal_type == \
                ParameterProposalType.MODEL_INFORMED:
            reconstructed_points = self.model.get_points_from_parameters(self.proposal.get_parameters())
            if self.batch_size == 1:
                reconstructed_points = reconstructed_points.unsqueeze(2)
            self.batch_mesh.set_points(reconstructed_points)
            self.batch_mesh.apply_rotation(self.proposal.get_rotation_parameters())
            self.batch_mesh.apply_translation(self.proposal.get_translation_parameters())
            self.batch_mesh.old_points = old_points
        elif parameter_proposal_type == ParameterProposalType.TRANSLATION:
            self.batch_mesh.apply_translation(-self.proposal.old_translation)
            self.batch_mesh.apply_translation(self.proposal.get_translation_parameters())
        else:
            self.batch_mesh.apply_translation(-self.proposal.get_translation_parameters())
            self.batch_mesh.apply_rotation(-self.proposal.old_rotation)
            self.batch_mesh.apply_rotation(self.proposal.get_rotation_parameters())
            self.batch_mesh.apply_translation(self.proposal.get_translation_parameters())
        self.batch_mesh.old_points = old_points
        self.points = self.batch_mesh.tensor_points

    def determine_quality(self, parameter_proposal_type: ParameterProposalType):
        """
        Initiates the calculation of the unnormalised posterior of the newly drawn sample.

        :param parameter_proposal_type: Specifies which parameters (model, translation or rotation) were drawn during
        the current iteration.
        :type parameter_proposal_type: ParameterProposalType
        """
        self.update_mesh(parameter_proposal_type)
        self.old_posterior = self.posterior
        if self.fixed_correspondences:
            correspondences = (batch_interpolate_barycentric_coords(self.batch_mesh.cells[0].data.unsqueeze(0).expand(self.batch_size, -1, -1), self.triangles.unsqueeze(0).expand(self.batch_size, -1), self.barycentric_coords.unsqueeze(0).expand(self.batch_size, -1, -1), self.points.permute(2, 0, 1))).permute(1, 2, 0)
            differences = torch.sub(self.target_points, correspondences)
            distances = torch.linalg.vector_norm(differences, dim=1)
            posterior = unnormalised_log_posterior(distances, self.inv_cov_operator, self.proposal.parameters,
                                                   self.proposal.translation, self.proposal.rotation, self.gamma,
                                                   self.var_like, self.var_prior_mod, self.uniform_pose_prior,
                                                   self.var_prior_trans, self.var_prior_rot)
            # posterior = unnormalised_log_posterior_curvature(distances, self.inv_cov_operator, self.proposal.parameters,
            #                                                  self.proposal.translation, self.proposal.rotation,
            #                                                  self.gamma, self.var_like, self.var_prior_mod,
            #                                                  self.mean_curv, self.uniform_pose_prior,
            #                                                  self.var_prior_trans, self.var_prior_rot)

        else:
            reference_meshes = self.batch_mesh.to_pytorch3d_meshes()
            # packed representation for faces
            verts_packed = reference_meshes.verts_packed()
            faces_packed = reference_meshes.faces_packed()
            tris = verts_packed[faces_packed]  # (T, 3, 3)
            tris_first_idx = reference_meshes.mesh_to_faces_packed_first_idx()
            # max_tris = reference_meshes.num_faces_per_mesh().max().item()

            points = self.target_clouds.points_packed()  # (P, 3)
            points_first_idx = self.target_clouds.cloud_to_packed_first_idx()
            max_points = self.target_clouds.num_points_per_cloud().max().item()

            # point to face squared distance: shape (P,)
            # requires dtype=torch.float32
            point_to_face_transposed = point_face_distance(points.float(), points_first_idx, tris.float(),
                                                           tris_first_idx, max_points).reshape(
                (-1, int(self.target.num_points)))
            point_to_face = torch.transpose(point_to_face_transposed, 0, 1)

            # Target is a point cloud, reference a mesh.
            posterior = unnormalised_log_posterior(torch.sqrt(point_to_face), self.inv_cov_operator,
                                                   self.proposal.parameters, self.proposal.translation,
                                                   self.proposal.rotation, self.gamma, self.var_like,
                                                   self.var_prior_mod, self.uniform_pose_prior, self.var_prior_trans,
                                                   self.var_prior_rot)
            # posterior = unnormalised_log_posterior_curvature(torch.sqrt(point_to_face), self.inv_cov_operator,
            #                                                  self.proposal.parameters, self.proposal.translation,
            #                                                  self.proposal.rotation, self.gamma, self.var_like,
            #                                                  self.var_prior_mod, self.mean_curv,
            #                                                  self.uniform_pose_prior, self.var_prior_trans,
            #                                                  self.var_prior_rot)

        self.posterior = posterior

    def decide(self, parameter_proposal_type: ParameterProposalType, full_target=None):
        """
        Implements the second and third step of the Metropolis algorithm (according to the lecture notes of the course
        "Statistical Shape Modelling" by Marcel Lüthi). Calculates the ratio of the unnormalised posterior values of the
        new and old samples. Decides which samples should form the current state based on the rules of the Metropolis
        algorithm. Parameter values (GaussianRandomWalkProposal) and mesh points (BatchMesh) are updated accordingly.

        :param parameter_proposal_type: Specifies which parameters (model, translation or rotation) were drawn during
        the current iteration.
        :type parameter_proposal_type: ParameterProposalType
        :param full_target: Complete actual target instance. Only needs to be specified if the residuals are to be
        saved.
        :type full_target: TorchMeshGpu
        """
        # log-ratio!
        ratio = torch.exp(self.posterior - self.old_posterior)
        probabilities = torch.min(ratio, torch.ones_like(ratio))
        randoms = torch.rand(self.batch_size, device=self.dev)
        decider = torch.gt(probabilities, randoms)
        # decider = torch.ones_like(decider)
        self.posterior = torch.where(decider, self.posterior, self.old_posterior)
        self.proposal.update(decider, self.posterior)
        self.batch_mesh.update_points(decider)
        self.points = self.batch_mesh.tensor_points
        if self.save_chain:
            self.full_chain.append(self.points)
        if self.save_residuals and full_target is not None:
            residual_c, residual_n = self.get_residual(full_target)
            self.residuals_c.append(residual_c)
            self.residuals_n.append(residual_n)

        if parameter_proposal_type == ParameterProposalType.MODEL_INFORMED:
            self.accepted_par += decider
            self.rejected_par += ~decider
        elif parameter_proposal_type == ParameterProposalType.MODEL_RANDOM:
            self.accepted_rnd += decider
            self.rejected_rnd += ~decider
        elif parameter_proposal_type == ParameterProposalType.TRANSLATION:
            self.accepted_trans += decider
            self.rejected_trans += ~decider
        else:
            self.accepted_rot += decider
            self.rejected_rot += ~decider

    def get_residual(self, full_target):
        """
        Calculates the distance from every point of the complete actual target to the current meshes.

        :param full_target: Complete actual target instance.
        :type full_target: TorchMeshGpu
        :return: Tuple containing 2 elements:
            - torch.Tensor: Distances from every point of the complete actual target to its corresponding point on the
            current meshes with shape (num_points, batch_size).
            - torch.Tensor: Distances from every point of the complete actual target to its closest point on the current
            meshes with shape (num_points, batch_size).
        :rtype: tuple
        """
        # TODO: To save runtime, do not recalculate the residuals if the sample was rejected, but use the values of the
        #  predecessor.
        differences = torch.sub(self.points, full_target.tensor_points.unsqueeze(-1))
        residual_c = torch.linalg.vector_norm(differences, dim=1)

        reference_meshes = self.batch_mesh.to_pytorch3d_meshes()
        if self.full_target_clouds is None:
            self.full_target_clouds = full_target.to_pytorch3d_pointclouds(batch_size=self.batch_size)
        # packed representation for faces
        verts_packed = reference_meshes.verts_packed()
        faces_packed = reference_meshes.faces_packed()
        tris = verts_packed[faces_packed]  # (T, 3, 3)
        tris_first_idx = reference_meshes.mesh_to_faces_packed_first_idx()
        # max_tris = reference_meshes.num_faces_per_mesh().max().item()

        points = self.full_target_clouds.points_packed()  # (P, 3)
        points_first_idx = self.full_target_clouds.cloud_to_packed_first_idx()
        max_points = self.full_target_clouds.num_points_per_cloud().max().item()

        # point to face squared distance: shape (P,)
        # requires dtype=torch.float32
        point_to_face_transposed = point_face_distance(points.float(), points_first_idx, tris.float(), tris_first_idx,
                                                       max_points).reshape((-1, int(full_target.num_points)))
        point_to_face = torch.transpose(point_to_face_transposed, 0, 1)
        residual_n = torch.sqrt(point_to_face)

        return residual_c, residual_n

    def acceptance_ratio(self, selector):
        # TODO: Handle the case where no ICP proposals are used. In this case, a division-by-zero error occurs.
        """
        Returns the ratio of accepted samples to the total number of (random) parameter draws.

        :param selector: Tensor of shape (batch_size,) that indicates which chains should be considered and which should
        not.
        :type selector: torch.Tensor
        :return: Tuple containing 5 float elements:
            - Ratio of accepted proposals when the model parameters have been adjusted using the informed ICP proposal.
            - Ratio of accepted proposals when the model parameters have been adjusted using a random walk proposal.
            - Ratio of accepted proposals when the translation parameters have been adjusted.
            - Ratio of accepted proposals when the rotation parameters have been adjusted.
            - Total ratio of accepted proposals.
        :rtype: tuple
        """
        accepted_tot = torch.sum(
            (self.accepted_par + self.accepted_rnd + self.accepted_trans + self.accepted_rot)[selector]).item()
        rejected_tot = torch.sum(
            (self.rejected_par + self.rejected_rnd + self.rejected_trans + self.rejected_rot)[selector]).item()
        ratio_par = float(torch.sum(self.accepted_par[selector]).item()) / torch.sum(
            (self.accepted_par + self.rejected_par)[selector]).item()
        ratio_rnd = float(torch.sum(self.accepted_rnd[selector]).item()) / torch.sum(
            (self.accepted_rnd + self.rejected_rnd)[selector]).item()
        ratio_trans = float(torch.sum(self.accepted_trans[selector]).item()) / torch.sum(
            (self.accepted_trans + self.rejected_trans)[selector]).item()
        ratio_rot = float(torch.sum(self.accepted_rot[selector]).item()) / torch.sum(
            (self.accepted_rot + self.rejected_rot)[selector]).item()
        ratio_tot = float(accepted_tot) / (accepted_tot + rejected_tot)
        return ratio_par, ratio_rnd, ratio_trans, ratio_rot, ratio_tot
        # return ratio_par, None, None, ratio_tot

    def change_device(self, dev):
        """
        Change the device on which the tensor operations are or will be allocated.

        :param dev: The future device on which the mesh data is to be saved.
        :type dev: torch.device
        """
        if self.dev == dev:
            return
        else:
            self.model.change_device(dev)
            self.proposal.change_device(dev)
            self.batch_mesh.change_device(dev)
            self.points = self.batch_mesh.tensor_points
            self.target.change_device(dev)
            self.target_points = self.target.tensor_points

            self.old_posterior = self.old_posterior.to(dev)
            self.posterior = self.posterior.to(dev)

            if self.save_chain and self.full_chain:
                self.full_chain = list(torch.stack(self.full_chain).to(dev).unbind(dim=0))

            if self.save_residuals and self.residuals_c and self.residuals_n:
                self.residuals_c = list(torch.stack(self.residuals_c).to(dev).unbind(dim=0))
                self.residuals_n = list(torch.stack(self.residuals_n).to(dev).unbind(dim=0))

            self.dev = dev

    def get_dict_chain_and_residuals(self):
        """
        Returns the chain with the meshes and the residuals as a dictionary, which can then be saved on disk. This needs
        a lot of memory.

        :return: Above-mentioned dictionary.
        :rtype: dict
        """
        if self.save_chain and self.save_residuals:
            return {'chain': self.full_chain, 'res_corr': self.residuals_c, 'res_clp': self.residuals_n}
        elif self.save_residuals:
            return {'res_corr': self.residuals_c, 'res_clp': self.residuals_n}
        else:
            warnings.warn("Warning: Chains and residuals were not saved.", UserWarning)


def create_target_aware_model(meshes, dev, target, part_target):
    gpa = ProcrustesAnalyser(meshes)
    gpa.generalised_procrustes_alignment()
    mean_shape = meshes[0].copy()
    mean_shape.set_points(torch.mean(gpa.points, dim=0), adjust_rotation_centre=True)
    icp = ICPAnalyser([part_target, mean_shape])
    icp.icp(swap_transformation=True)
    target.set_points(
        (torch.matmul(target.tensor_points, icp.transformation[0]) + icp.transformation[1].unsqueeze(1)).squeeze(),
        adjust_rotation_centre=True)

    # Estimate the correspondences using model sampling
    closest_pts, dists, _, _ = estimate_correspondences(mean_shape, part_target, dev)

    # Remove wrong correspondences by detecting the nearest points located on the mesh boundary
    # boundary_edges = trimesh.base.Trimesh(mesh_cpu, mesh_faces, process=True, validate=True).outline().vertex_nodes
    boundary_edges = extract_bounds_verts(part_target.cells[0].data, extract_edges=True)
    observed = ~(
        is_point_on_line_segment(closest_pts, part_target.tensor_points[boundary_edges],
                                 tol=1e-2))

    # Align all meshes using the observed points
    meshes_observed_verts = copy.deepcopy(meshes)
    _ = list(map(lambda x, y: x.set_points(y.tensor_points[observed, :]), meshes_observed_verts, meshes))
    target_corresponding_verts = part_target.copy()
    target_corresponding_verts.set_points(closest_pts[observed])
    meshes_observed_verts.insert(0, target_corresponding_verts)
    gpa_target = ProcrustesAnalyser(meshes_observed_verts)
    gpa_target.generalised_procrustes_alignment()

    # Apply the found transformations to full meshes
    transformations = get_transformation_matrix_from_rot_and_trans(gpa_target.transformation[0].permute(1, 2, 0),
                                                                   gpa_target.transformation[1].permute(1, 0))
    meshes.insert(0, part_target)
    _ = list(map(lambda x, y: x.apply_transformation(y), meshes, transformations.permute(2, 0, 1)))
    target.apply_transformation(transformations.permute(2, 0, 1)[0])
    del meshes[0]

    # Estimate the correspondences using target sampling
    _, _, fid, bc = estimate_correspondences(part_target, mean_shape, dev)
    model = PointDistributionModel(meshes=meshes)
    return model, fid, bc


def create_target_unaware_model(meshes, dev, target):
    ProcrustesAnalyser(meshes).generalised_procrustes_alignment()
    model = PointDistributionModel(meshes=meshes)
    mean_shape = meshes[0].copy()
    mean_shape.set_points(model.get_mean_points(), adjust_rotation_centre=True)
    ICPAnalyser([target, mean_shape]).icp()
    model.mean = mean_shape.tensor_points.reshape(3 * model.num_points, 1)

    # Estimate the correspondences using target sampling
    _, _, fid, bc = estimate_correspondences(target, mean_shape, dev)

    return model, fid, bc


def estimate_correspondences(x, y, dev):
    cloud_cpu = x.tensor_points.cpu().numpy()
    mesh_cpu = y.tensor_points.cpu().numpy()
    mesh_faces = y.cells[0].data.cpu().numpy()
    dists, fid, bc = pcu.closest_points_on_mesh(cloud_cpu, mesh_cpu, mesh_faces)
    closest_pts = pcu.interpolate_barycentric_coords(mesh_faces, fid, bc, mesh_cpu)
    return torch.tensor(closest_pts, device=dev), torch.tensor(dists, device=dev), torch.tensor(fid, device=dev), \
        torch.tensor(bc, device=dev)


def interpolate_barycentric_coords(f, fi, bc, attribute):
    f_vertices = f[fi]
    f_attributes = attribute[f_vertices]
    return (f_attributes * bc.unsqueeze(-1)).sum(dim=1)


def batch_interpolate_barycentric_coords(f, fi, bc, attribute):
    batch_size = f.shape[0]
    batch_indices = torch.arange(batch_size, device=f.device).unsqueeze(-1)
    batch_indices_fi = batch_indices.expand(-1, fi.shape[1])
    f_vertices = f[batch_indices_fi, fi]
    batch_indices_f = batch_indices.unsqueeze(-1).expand(-1, f_vertices.shape[1], f_vertices.shape[2])
    f_attributes = attribute[batch_indices_f, f_vertices]
    return (f_attributes * bc.unsqueeze(-1)).sum(dim=2)