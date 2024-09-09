import math
import torch
import trimesh.base
from torch.autograd.function import once_differentiable
from trimesh import Trimesh
from pytorch3d import _C
from pytorch3d.loss.point_mesh_distance import _PointFaceDistance, point_face_distance

from src.mesh.TMesh import BatchTorchMesh
from src.model.PointDistribution import PointDistributionModel, BatchedPointDistributionModel
from src.sampling.proposals.GaussRandWalk import ParameterProposalType, GaussianRandomWalkProposal


def is_point_on_line_segment(points, boundary_vertices, tol=1e-6):
    """
    Checks if given points are on one of some line segments defined by a set of interconnected boundary vertices.
    This method can be used to test whether determined closest points are located on the border of the (partial,
    non-watertight) target mesh.

    :param points: The points to check with shape (N, 3).
    :type points: torch.Tensor
    :param boundary_vertices: The boundary vertices with shape (M, 2, 3).
    :type boundary_vertices: torch.Tensor
    :param tol: Tolerance for floating-point comparisons.
    :type tol: float
    :return: Boolean indicating whether the points are on the line segments with shape (N,).
    :rtype: torch.Tensor
    """
    boundary_1, boundary_2 = boundary_vertices[:, 0, :], boundary_vertices[:, 1, :]
    boundary_1, boundary_2 = boundary_1.unsqueeze(0).unsqueeze(2), boundary_2.unsqueeze(0).unsqueeze(2)
    points = points.unsqueeze(1).unsqueeze(1)

    line = boundary_2 - boundary_1
    vector = points - boundary_1

    # 1: Check if the points are collinear with the line segment
    cross = torch.cross(line, vector, dim=-1)
    res_1 = torch.linalg.norm(cross, dim=-1) > tol

    # 2: Check if the points are within the bounds of the segment
    dot = torch.sum(vector * line, dim=-1)
    res_2 = dot < 0

    line_norm = torch.sum(line * line, dim=-1)
    res_3 = dot > line_norm

    return torch.any(~(res_1 | res_2 | res_3), dim=1).squeeze()


def point_face_distance(points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area=float(5e-3)):
    """
    Function to use the extended version of the point-face distance calculation.
    """
    return ExtendedPointFaceDistance.apply(
        points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area
    )


class ExtendedPointFaceDistance(_PointFaceDistance):
    """
    Subclass of _PointFaceDistance that overrides the forward method to return both dists and idxs.
    """

    @staticmethod
    def forward(
            ctx,
            points,
            points_first_idx,
            tris,
            tris_first_idx,
            max_points,
            min_triangle_area=float(5e-3),
    ):
        dists, idxs = _C.point_face_dist_forward(
            points,
            points_first_idx,
            tris,
            tris_first_idx,
            max_points,
            min_triangle_area,
        )
        ctx.save_for_backward(points, tris, idxs)
        ctx.min_triangle_area = min_triangle_area

        return dists, idxs

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists):
        grad_dists = grad_dists.contiguous()
        points, tris, idxs = ctx.saved_tensors
        min_triangle_area = ctx.min_triangle_area
        grad_points, grad_tris = _C.point_face_dist_backward(
            points, tris, idxs, grad_dists, min_triangle_area
        )
        return grad_points, None, grad_tris, None, None, None


class ClosestPointProposal(GaussianRandomWalkProposal):

    def __init__(self, batch_size, starting_parameters, dev, reference, batched_reference, target, model,
                 sigma_mod=1.0, sigma_trans=10.0, sigma_rot=0.01, chain_length_step=1000):
        """
        The class is used to draw new values for the parameters. The class supports three types of parameter: Model
        parameters, translation and rotation. It is designed for batches. All parameters are therefore always generated
        for an entire batch of proposals.
        This is an informed proposal. The main idea comes from the paper by D. Madsen et al.: "A Closest Point Proposal
        for MCMC-based Probabilistic Surface Registration" and uses a posterior model based on estimated correspondences
        to propose randomized informed samples.
        However, a simplified version is implemented here. The posterior model is only calculated once at the beginning.
        A random walk is then performed in this posterior space and the guessed parameters are projected into the prior
        parameter space.
        Furthermore, to keep the proposal distribution symmetrical, the posterior is centred on the current state. (In
        this case, the current state equals the starting state, as the posterior model is only calculated once at the
        beginning.) Here, it is assumed that the start state is the same for all elements of the batch. All elements of
        the batch work with the exact same posterior model.
        All types of parameters (posterior model and pose) are drawn independently of a Gaussian distribution with mean
        value at the previous parameter value and a standardised variance. The variance is defined separately for all 3
        types of parameters.

        :param batch_size: Number of proposals to be generated simultaneously.
        :type batch_size: int
        :param starting_parameters: Start values of the model parameters. When new parameters are drawn for the first
        time, these initial values are used as the mean value of the Gaussian distribution (for all the elements of the
        batch). The shape of the tensor is assumed to be (num_model_parameters,).
        :type starting_parameters: torch.Tensor
        :param dev: An object representing the device on which the tensor operations are or will be allocated.
        :type dev: torch.device
        :param reference: Single mesh instance. Can contain any points, as these are overwritten during the calculation
        of the analytic posterior.
        :type reference: TorchMeshGpu
        :param batched_reference: Current batch with mesh instances generated from the starting parameters. When the
        analytic posterior is calculated, the distances to the target are determined. As the model parameters are the
        same for all elements of the batch, the mesh points must also be the same for every element of the batch.
        :type batched_reference: BatchTorchMesh
        :param target: Instance to be fitted. Necessary for calculating the analytic posterior. Remark: It is required
        that the target has the same ‘batch_size’ as ‘batch_reference’.
        :type target: BatchTorchMesh
        :param model: Prior model, which serves as the starting point for calculating the posterior model.
        :type model: PointDistributionModel
        :param sigma_mod: Variance of the model parameters. Note: May be set larger for this type of proposal compared
        to the uninformed random walk proposals.
        :type sigma_mod: float
        :param sigma_trans: Variance of the translation parameters.
        :type sigma_trans: float
        :param sigma_rot: Variance of the rotation parameters.
        :type sigma_rot: float
        :param chain_length_step: Step size with which the Markov chain is extended. If chain_length_step many
        samples were drawn, the chain must be extended again accordingly.
        :type chain_length_step: int
        """
        super().__init__(batch_size, starting_parameters, dev, sigma_mod, sigma_trans, sigma_rot, chain_length_step)
        self.posterior_parameters = torch.zeros(model.rank, 1, device=self.dev).repeat(1, self.batch_size)
        self.reference = reference.copy()
        self.target = target
        self.partial_target = self.target.num_points < self.reference.num_points
        # self.target = BatchTorchMesh(target, target.id, batch_size=1)
        self.prior_model = model
        self.posterior_model, self.projection_matrix = self.calculate_posterior_model(batched_reference)
        self.old_posterior_parameters = None

    def calculate_posterior_model(self, batched_reference, sigma_n=3.0, sigma_v=100.0):
        """
        Executes the calculation of the posterior model.
        In contrast to the algorithm in the paper by D. Madsen et al., all points on the selected current model instance
        are used as observations.
        The step-length mentioned in step 6 is set to 1.0 (currently without the possibility of adjustment). With a step
        size of 1.0, the proposed sample is an independent sample from the calculated posterior.
        The noise of the observations is modeled with low variance along the normal direction and high variance along
        the surface. For this reason, the corresponding facet and vertex normals must be calculated along the way.
        Remark: Currently only designed to calculate the posterior at the beginning of the sampling process and not
        optimised for calculation on a GPU. Uses Trimesh, which does not support GPUs.

        :param batched_reference: Current batch with mesh instances generated from the starting parameters. When the
        analytic posterior is calculated, the distances to the target are determined. As the model parameters are the
        same for all elements of the batch, the mesh points must also be the same for every element of the batch.
        :type batched_reference: BatchTorchMesh
        :param sigma_n: Variance along the normal direction.
        :type sigma_n: float
        :param sigma_v: Variance along the surface.
        :type sigma_v: float
        :return: Tuple with 2 elements:
            - PointDistributionModel: Calculated point distribution model (PDM).
            - torch.Tensor: Projection matrix with which the posterior model parameters can be projected into the
              prior model space. Tensor with shape (num_rank, num_rank).
        :rtype: tuple
        """
        m = self.prior_model.num_points
        reconstructed_points = batched_reference.tensor_points[:, :, 0].squeeze()
        self.reference.set_points(reconstructed_points, reset_com=True)

        # 1: Sample m points {s_i} on the current model instance \Gamma[\alpha].
        # sampled_indexes = torch.tensor(np.random.choice(np.arange(0, self.prior_model.num_points), m, replace=False),
        #                               dtype=torch.int64)

        # 2: For every point s_i, i \in [0, ..., m] find the closest point c_i on the target \Gamma_T.
        # Trimesh doesn't support GPU operations yet. Since this class is designed to calculate only one posterior model
        # at the beginning of the Markov chain, this runtime disadvantage is negligible.

        target_mesh = Trimesh(self.target.tensor_points[:, :, 0].cpu(), self.target.cells[0].data.cpu())
        closest_points, _, _ = target_mesh.nearest.on_surface(self.reference.tensor_points.cpu())
        closest_points = torch.tensor(closest_points, dtype=torch.float, device=self.dev)

        # All points from the reference are mapped to the closest point of the target, even when they are not observed.
        # To counter this, all predicted correspondences, where the predicted target point is part of the target
        # surface's boundary, are removed.
        on_boundary = torch.zeros(m, dtype=torch.bool, device=self.dev)

        if self.partial_target:
            outline = target_mesh.outline().entities
            for line in outline:
                if isinstance(line, trimesh.path.entities.Line):
                    boundary_vertices = self.target.tensor_points[:, :, 0][line.nodes]
                    on_boundary = is_point_on_line_segment(closest_points, boundary_vertices)

        # 3: Construct the set of observations L based on corresponding landmark pairs (s_i, c_i) according to eq. 9
        # and define the noise \epsilon_i \distas \mathcal{N}(0, \Sigma_{s_{i}} using eq. 16
        # differences = closest_points - sampled_points
        self.reference.calc_facet_normals()
        mean_vertex_normals = torch.tensor(trimesh.geometry.mean_vertex_normals(self.prior_model.num_points,
                                                                                self.reference.cells[0].data.cpu(),
                                                                                self.reference.cell_data[
                                                                                    "facet_normals"][0].data.cpu()),
                                           dtype=torch.float, device=self.dev)
        mean_vertex_normals = mean_vertex_normals / torch.norm(mean_vertex_normals, dim=1, keepdim=True)

        # Calculate 2 perpendicular vectors for each normal to define the \Sigma_{s_{i}}.
        # Select a reference vector that is not parallel to most of the normals.
        ref = torch.tensor([1.0, 0.0, 0.0], device=self.dev)

        # To avoid issues where the normal might be parallel to the reference vector
        # we create a matrix to select the next-best vector for those cases.
        parallel_mask = (torch.abs(mean_vertex_normals @ ref) > 0.9).unsqueeze(1)
        alt = torch.tensor([0.0, 1.0, 0.0], device=self.dev)
        ref = torch.where(parallel_mask, alt, ref)

        # Compute the first tangent vector by taking the cross product of the normal and the reference vector.
        v1 = torch.cross(mean_vertex_normals, ref.expand_as(mean_vertex_normals), dim=1)
        v1 = v1 / torch.norm(v1, dim=1, keepdim=True)

        # Compute the second tangent vector by taking the cross product of the normal and the first tangent.
        v2 = torch.cross(mean_vertex_normals, v1, dim=1)
        v2 = v2 / torch.norm(v2, dim=1, keepdim=True)

        directions = torch.stack((mean_vertex_normals, v1, v2), dim=2)
        sigma = torch.diag(torch.tensor([sigma_n, sigma_v, sigma_v], device=self.dev))
        covariances = directions @ sigma @ directions.transpose(1, 2)

        # 4: Compute the analytic posterior \mathcal{M}_{\alpha} (eq. 10) with L and {\Sigma_{s_{i}}}.
        mean_prior = self.reference.tensor_points.reshape((3 * m, -1))
        cov_prior = self.prior_model.get_covariance()

        cov_reshaped = torch.zeros((3 * m, 3 * m), device=self.dev)
        indices = torch.arange(0, 3 * m, 3, device=self.dev).unsqueeze(1) + torch.arange(3, device=self.dev).unsqueeze(
            0)
        indices = indices[~on_boundary]
        cov_reshaped[indices.unsqueeze(-1), indices.unsqueeze(-2)] += covariances[~on_boundary]

        K_yy_inv = torch.inverse(
            cov_prior[indices.flatten()][:, indices.flatten()] + cov_reshaped[indices.flatten()][:,
                                                                 indices.flatten()])
        K_xy = cov_prior[:, indices.flatten()]
        cov_posterior = cov_prior - K_xy @ K_yy_inv @ torch.t(K_xy)
        posterior_model = PointDistributionModel(mean_and_cov=True, mean=mean_prior, cov=cov_posterior,
                                                 rank=self.prior_model.rank)
        projection_matrix = torch.diag(1 / torch.sqrt(
            self.prior_model.get_eigenvalues())) @ self.prior_model.eigenvectors.T @ posterior_model.get_components()
        return posterior_model, projection_matrix

    def propose(self, parameter_proposal_type: ParameterProposalType):
        """
        Updates the parameter values using a Gaussian random walk in posterior space (see class description).

        :param parameter_proposal_type: Specifies which parameters are to be drawn.
        :type parameter_proposal_type: ParameterProposalType
        """
        if parameter_proposal_type == ParameterProposalType.MODEL:
            perturbations = torch.randn((self.num_parameters, self.batch_size), device=self.dev)
        else:
            perturbations = torch.randn((3, self.batch_size), device=self.dev)

        self.old_parameters = self.parameters
        self.old_posterior_parameters = self.posterior_parameters
        self.old_translation = self.translation
        self.old_rotation = self.rotation

        if parameter_proposal_type == ParameterProposalType.MODEL:
            self.posterior_parameters = self.posterior_parameters + perturbations * self.sigma_mod
            self.parameters = self.projection_matrix @ self.posterior_parameters
        elif parameter_proposal_type == ParameterProposalType.TRANSLATION:
            self.translation = self.translation + perturbations * self.sigma_trans * math.sqrt(
                (1 / self.prior_model.num_points))
        else:
            self.rotation = self.rotation + perturbations * self.sigma_rot
        # self.parameters = torch.zeros((self.num_parameters, self.batch_size))
        # self.parameters = perturbations * self.sigma_mod

    def update_parameters(self, decider):
        """
        Internal method that updates the parameters and the log-density values of the posterior according to the
        information from the passed decider.

        :param decider: Boolean tensor of the shape (batch_size,), which indicates for each element of the batch whether
        the new parameter values were accepted (=True) or rejected (=False).
        :type decider: torch.Tensor
        """
        self.parameters = torch.where(decider.unsqueeze(0), self.parameters,
                                      self.old_parameters)
        self.posterior_parameters = torch.where(decider.unsqueeze(0), self.posterior_parameters,
                                                self.old_posterior_parameters)
        self.translation = torch.where(decider.unsqueeze(0), self.translation, self.old_translation)
        self.rotation = torch.where(decider.unsqueeze(0), self.rotation, self.old_rotation)
        self.old_parameters = None
        self.old_posterior_parameters = None
        self.old_translation = None
        self.old_rotation = None

    def change_device(self, dev):
        """
        Change the device on which the tensor operations are or will be allocated. Only execute between completed
        iterations of the sampling process.

        :param dev: The future device on which the mesh data is to be saved.
        :type dev: torch.device
        """
        if self.dev == dev:
            return
        else:
            self.parameters = self.parameters.to(dev)
            self.translation = self.translation.to(dev)
            self.rotation = self.rotation.to(dev)
            self.chain = self.chain.to(dev)
            self.posterior = self.posterior.to(dev)

            self.posterior_parameters = self.posterior_parameters.to(dev)
            self.reference.change_device(dev)
            self.target.change_device(dev)
            self.prior_model.change_device(dev)
            self.posterior_model.change_device(dev)
            self.projection_matrix = self.projection_matrix.to(dev)

            self.dev = dev


class FullClosestPointProposal(GaussianRandomWalkProposal):

    def __init__(self, batch_size, starting_parameters, dev, batched_reference, target, model, sigma_mod=1.0,
                 sigma_trans=10.0, sigma_rot=0.01, recalculation_period=100, chain_length_step=1000):
        """
        The class is used to draw new values for the parameters. The class supports three types of parameter: Model
        parameters, translation and rotation. It is designed for batches. All parameters are therefore always generated
        for an entire batch of proposals.
        This is an informed proposal. The main idea comes from the paper by D. Madsen et al.: "A Closest Point Proposal
        for MCMC-based Probabilistic Surface Registration" and uses a posterior model based on estimated correspondences
        to propose randomized informed samples.
        The posterior model is determined separately for each element of the batch. The posterior models are
        recalculated regularly. How often can be determined using the corresponding input parameter
        'recalculation_period'. A random walk is then performed in this posterior space and the guessed parameters are
        projected into the prior parameter space.
        To keep the proposal distribution symmetrical, the posterior is centred on the current state.
        All types of parameters (posterior model and pose) are drawn independently of a Gaussian distribution with mean
        value at the previous parameter value and a standardised variance. The variance is defined separately for all 3
        types of parameters.

        :param batch_size: Number of proposals to be generated simultaneously.
        :type batch_size: int
        :param starting_parameters: Start values of the prior model parameters. When new parameters are drawn for the
        first time, these initial values are used as the mean value of the Gaussian distribution. The shape of the
        tensor is assumed to be (num_model_parameters, batch_size).
        :type starting_parameters: torch.Tensor
        :param dev: An object representing the device on which the tensor operations are or will be allocated.
        :type dev: torch.device
        :param batched_reference: Meshes that are generated from the initial model parameters.
        :type batched_reference: BatchTorchMesh
        :param target: Instance to be fitted for each batch. Normally, it is the same shape for all elements of the
        batch. Necessary for calculating the analytic posterior.
        :type target: BatchTorchMesh
        :param model: Prior model, which serves as the starting point for calculating the posterior model.
        :type model: BatchedPointDistributionModel
        :param sigma_mod: Variance of the model parameters. Note: May be set larger for this type of proposal compared
        to the uninformed random walk proposals.
        :type sigma_mod: float
        :param sigma_trans: Variance of the translation parameters.
        :type sigma_trans: float
        :param sigma_rot: Variance of the rotation parameters.
        :type sigma_rot: float
        :param recalculation_period: The parameter determines after how many iterations the posterior model is
        recalculated. With the default value of 1, a new posterior model is calculated in each iteration.
        :type recalculation_period: int
        :param chain_length_step: Step size with which the Markov chain is extended. If chain_length_step many
        samples were drawn, the chain must be extended again accordingly.
        :type chain_length_step: int
        """
        super().__init__(batch_size, starting_parameters[:, 0], dev, sigma_mod, sigma_trans, sigma_rot,
                         chain_length_step)
        self.parameters = starting_parameters
        self.posterior_parameters = torch.zeros(model.rank, 1, device=self.dev).expand(-1, self.batch_size)
        self.batched_reference = batched_reference
        self.target = target
        self.partial_target = self.target.num_points < self.batched_reference.num_points
        # self.target = BatchTorchMesh(target, target.id, batch_size=1)
        self.prior_model = model
        self.posterior_model, self.projection_matrix, self.old_posterior_parameters = None, None, None
        self.recalculation_period = recalculation_period
        self.counter = 0

    def calculate_posterior_model(self, batched_reference, sigma_n=3.0, sigma_v=100.0):
        """
        Executes the calculation of the posterior model.
        In contrast to the algorithm in the paper by D. Madsen et al., all points on the current model instances
        are used as observations.
        The step-length mentioned in step 6 is set to 1.0 (currently without the possibility of adjustment). With a step
        size of 1.0, the proposed sample is an independent sample from the calculated posterior.
        The noise of the observations is modeled with low variance along the normal direction and high variance along
        the surface. For this reason, the corresponding vertex normals must be calculated along the way.

        :param batched_reference: Meshes that are generated from the current model parameters.
        :type batched_reference: BatchTorchMesh
        :param sigma_n: Variance along the normal direction.
        :type sigma_n: float
        :param sigma_v: Variance along the surface.
        :type sigma_v: float
        :return: Tuple with 2 elements:
            - BatchedPointDistributionModel: Calculated point distribution models. A separate posterior PDM is
              calculated for each element of the batch.
            - torch.Tensor: Projection matrices with which the posterior model parameters can be projected into the
              prior model space. Tensor with shape (num_rank, num_rank, batch_size).
        :rtype: tuple
        """
        self.posterior_parameters = torch.zeros(self.prior_model.rank, 1, device=self.dev).expand(-1, self.batch_size)
        m = self.prior_model.num_points
        reconstructed_points = batched_reference.tensor_points

        # 1: Sample m points {s_i} on the current model instance \Gamma[\alpha].
        # In this implementation, all the points on \Gamma[\alpha] are chosen.

        # 2: For every point s_i, i \in [0, ..., m] find the closest point c_i on the target \Gamma_T.
        target_mesh = Trimesh(self.target.tensor_points[:, :, 0].cpu(), self.target.cells[0].data.cpu())

        # Alternative calculation using Pytorch3d, which directly returns the squared distance to the next point c_i on
        # \Gamma_T and the index of the triangle on which this point lies.
        reference_clouds = self.batched_reference.to_pytorch3d_pointclouds()
        target_meshes = self.target.to_pytorch3d_meshes()
        reference_meshes = self.batched_reference.to_pytorch3d_meshes()

        verts_packed = target_meshes.verts_packed()
        faces_packed = target_meshes.faces_packed()
        tris = verts_packed[faces_packed]  # (T, 3, 3)
        tris_first_idx = target_meshes.mesh_to_faces_packed_first_idx()

        points = reference_clouds.points_packed()  # (P, 3)
        points_first_idx = reference_clouds.cloud_to_packed_first_idx()
        max_points = reference_clouds.num_points_per_cloud().max().item()

        # point to face squared distance: shape (P,)
        # requires dtype=torch.float32
        point_to_face, triangle_idx = point_face_distance(points.float(), points_first_idx, tris.float(),
                                                          tris_first_idx, max_points)
        triangle_idx = triangle_idx.reshape((self.batch_size, -1)).t() - tris_first_idx
        # All points from the reference are mapped to the closest point of the target, even when they are not observed.
        # To counter this, all predicted correspondences, where the predicted target point is part of a triangle at the
        # target surface's boundary, are removed.
        on_boundary = torch.zeros((m, self.batch_size), dtype=torch.bool, device=self.dev)

        if self.partial_target:
            boundary_verts = torch.tensor(target_mesh.outline().referenced_vertices, device=self.dev)
            boundary_mask = (self.target.cells[0].data.unsqueeze(1) == boundary_verts.unsqueeze(0).unsqueeze(-1))
            boundary_tris = boundary_mask.any(-1).any(-1).nonzero(as_tuple=True)[0]
            on_boundary = (boundary_tris.unsqueeze(0).unsqueeze(0) == triangle_idx.unsqueeze(-1)).any(-1)

        # 3: Construct the set of observations L based on corresponding landmark pairs (s_i, c_i) according to eq. 9
        # and define the noise \epsilon_i \distas \mathcal{N}(0, \Sigma_{s_{i}} using eq. 16
        # (Almost the same code as in ClosestPointProposal.)
        mean_vertex_normals = reference_meshes.verts_normals_padded().permute(1, 2, 0)
        ref = torch.tensor([1.0, 0.0, 0.0], device=self.dev).repeat(m, self.batch_size, 1).transpose(1, 2)

        parallel_mask = torch.abs(torch.sum(mean_vertex_normals * ref, dim=1)) > 0.9
        ref.transpose(1, 2)[parallel_mask] = torch.tensor([0.0, 1.0, 0.0], device=self.dev)
        v1 = torch.cross(mean_vertex_normals, ref, dim=1)
        v1 = v1 / torch.norm(v1, dim=1, keepdim=True)
        v2 = torch.cross(mean_vertex_normals, v1, dim=1)
        v2 = v2 / torch.norm(v2, dim=1, keepdim=True)

        directions = torch.stack((mean_vertex_normals, v1, v2), dim=3).permute(0, 2, 1, 3)
        sigma = torch.diag(torch.tensor([sigma_n, sigma_v, sigma_v], device=self.dev)).view(1, 1, 3, 3)
        covariances = torch.matmul(torch.matmul(directions, sigma), directions.transpose(2, 3))

        # 4: Compute the analytic posterior \mathcal{M}_{\alpha} (eq. 10) with L and {\Sigma_{s_{i}}}.
        cov_prior = self.prior_model.get_covariance()

        cov_reshaped = torch.zeros((3 * m, 3 * m, self.batch_size), device=self.dev)
        covariances = covariances * (~on_boundary.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 3))

        idx = torch.arange(0, 3 * m, 3, device=self.dev).unsqueeze(1) + torch.arange(3, device=self.dev).unsqueeze(0)
        x_idx, y_idx = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.batch_size), idx.unsqueeze(-2).unsqueeze(
            -1).expand(-1, -1, -1, self.batch_size)
        batch_idx = torch.arange(self.batch_size).to(self.dev).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        cov_reshaped[x_idx, y_idx, batch_idx] += covariances.permute(0, 2, 3, 1)

        K_xy_batch = torch.zeros((3 * m, 3 * torch.max(torch.sum(~on_boundary, dim=0)), self.batch_size),
                                 device=self.dev)
        K_yy_batch = torch.zeros((3 * torch.max(torch.sum(~on_boundary, dim=0)),
                                  3 * torch.max(torch.sum(~on_boundary, dim=0)), self.batch_size), device=self.dev)
        observed_idx = torch.nonzero(~on_boundary, as_tuple=False)
        y2_idx = torch.stack([3 * observed_idx[:, 0], 3 * observed_idx[:, 0] + 1, 3 * observed_idx[:, 0] + 2], dim=1).flatten()
        batch2_idx = observed_idx[:, 1].repeat_interleave(3)
        K_xy_gather = cov_prior[:, y2_idx, batch2_idx]
        cov_reshaped += cov_prior
        K_yy_gather = cov_reshaped[:, y2_idx, batch2_idx]

        for i in range(self.batch_size):
            K_xy_vec = K_xy_gather[:, batch2_idx == i]
            K_xy_batch[:, :K_xy_vec.size()[1], i] = K_xy_vec
            K_yy_batch[:, :, i] = K_yy_gather[:, batch2_idx == i][y2_idx[batch2_idx == i], :]

        K_yy_inv = torch.inverse(K_yy_batch.permute(2, 0, 1))
        # TODO: There is a faster method to calculate the posterior model that takes advantage of the fact that the
        #       model is only of low rank. This is implemented in Scalismo.
        mean_posterior = reconstructed_points.reshape((3 * m, self.batch_size))
        cov_posterior = (cov_prior.permute(2, 0, 1) - K_xy_batch.permute(2, 0, 1) @ K_yy_inv @ K_xy_batch
                         .permute(2, 1, 0)).permute(1, 2, 0)
        posterior_model = BatchedPointDistributionModel(mean_and_cov=True, mean=mean_posterior, cov=cov_posterior,
                                                        rank=self.prior_model.rank, batch_size=self.batch_size)
        prior_eigenvalues = self.prior_model.get_eigenvalues().T
        prior_eigenvectors = self.prior_model.eigenvectors.permute(2, 1, 0)
        posterior_components = posterior_model.get_components().permute(2, 0, 1)
        projection_matrices = torch.diag_embed(
            1 / torch.sqrt(prior_eigenvalues)) @ prior_eigenvectors @ posterior_components
        return posterior_model, projection_matrices.permute(1, 2, 0)

    def propose(self, parameter_proposal_type: ParameterProposalType, **kwargs):
        """
        Updates the parameter values using a Gaussian random walk in posterior space (see class description).

        :param parameter_proposal_type: Specifies which parameters are to be drawn.
        :type parameter_proposal_type: ParameterProposalType
        :param kwargs: Additional keyword arguments that include:
            - batch_mesh (BatchTorchMesh): Meshes that are generated from the current model parameters.
        :type kwargs: dict
        """

        if self.counter % self.recalculation_period == 0:
            self.posterior_model, self.projection_matrix = self.calculate_posterior_model(
                kwargs.get('batch_mesh', self.batched_reference))

        if parameter_proposal_type == ParameterProposalType.MODEL:
            perturbations = torch.randn((self.num_parameters, self.batch_size), device=self.dev)
        else:
            perturbations = torch.randn((3, self.batch_size), device=self.dev)

        self.old_parameters = self.parameters
        self.old_posterior_parameters = self.posterior_parameters
        self.old_translation = self.translation
        self.old_rotation = self.rotation

        if parameter_proposal_type == ParameterProposalType.MODEL:
            self.posterior_parameters = self.posterior_parameters + perturbations * self.sigma_mod
            self.parameters = self.projection_matrix.permute(2, 0, 1) @ self.posterior_parameters.T.unsqueeze(-1)
            self.parameters = self.parameters.squeeze().T
        elif parameter_proposal_type == ParameterProposalType.TRANSLATION:
            self.translation = self.translation + perturbations * self.sigma_trans * math.sqrt(
                (1 / self.prior_model.num_points))
        else:
            self.rotation = self.rotation + perturbations * self.sigma_rot

        self.counter += 1

    def update_parameters(self, decider):
        """
        Internal method that updates the parameters and the log-density values of the posterior according to the
        information from the passed decider.

        :param decider: Boolean tensor of the shape (batch_size,), which indicates for each element of the batch whether
        the new parameter values were accepted (=True) or rejected (=False).
        :type decider: torch.Tensor
        """
        self.parameters = torch.where(decider.unsqueeze(0), self.parameters,
                                      self.old_parameters)
        self.posterior_parameters = torch.where(decider.unsqueeze(0), self.posterior_parameters,
                                                self.old_posterior_parameters)
        self.translation = torch.where(decider.unsqueeze(0), self.translation, self.old_translation)
        self.rotation = torch.where(decider.unsqueeze(0), self.rotation, self.old_rotation)
        self.old_parameters = None
        self.old_posterior_parameters = None
        self.old_translation = None
        self.old_rotation = None

    def change_device(self, dev):
        """
        Change the device on which the tensor operations are or will be allocated. Only execute between completed
        iterations of the sampling process.

        :param dev: The future device on which the mesh data is to be saved.
        :type dev: torch.device
        """
        if self.dev == dev:
            return
        else:
            self.parameters = self.parameters.to(dev)
            self.translation = self.translation.to(dev)
            self.rotation = self.rotation.to(dev)
            self.chain = self.chain.to(dev)
            self.posterior = self.posterior.to(dev)

            self.posterior_parameters = self.posterior_parameters.to(dev)
            self.batched_reference.change_device(dev)
            self.target.change_device(dev)
            self.prior_model.change_device(dev)
            self.posterior_model.change_device(dev)
            self.projection_matrix = self.projection_matrix.to(dev)

            self.dev = dev
