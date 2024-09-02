import math

import torch
from pytorch3d.loss.point_mesh_distance import point_face_distance
from torch.distributions import Uniform

from src.sampling.proposals.GaussRandWalk import ParameterProposalType


def unnormalised_log_posterior(distances, parameters, translation, rotation, sigma_lm, sigma_prior, dev):
    """
    Calculates the unnormalised log posterior given the distance of each point of the target to the closest point on the
    surface of the current reference and the model parameters of the current reference.
    This method can also be called for entire batches of references.
    The prior term pushes the solution towards a more likely shape by penalizing unlikely shape deformations. A uniform
    prior is assumed for translation and rotation. For the likelihood term, the L2 distance (independent point evaluator
    likelihood) is used. The L2 distance is evaluated using a zero-mean Gaussian with indicated variance.

    :param distances: Tensor with distances of each point of the target to the closest point on the surface of the
    references considered with shape (num_target_points, batch_size).
    :type distances: torch.Tensor
    :param parameters: Tensor with the current model parameters of the references under consideration with shape
    (num_parameters, batch_size).
    :type parameters: torch.Tensor
    :param translation: Tensor with the current translation parameters of the references under consideration with shape
    (3, batch_size).
    :type translation: torch.Tensor
    :param rotation: Tensor with the current rotation parameters of the references under consideration with shape
    (3, batch_size).
    :type rotation: torch.Tensor
    :param sigma_lm: Variance of the zero-mean Gaussian, which is used to evaluate the L2 distance. The same variance
    value is used for all 3 dimensions (isotropic distribution).
    :type sigma_lm: float
    :param sigma_prior: Variance of the model parameters (prior term calculation).
    :type sigma_prior: float
    :return: Tensor with unnormalised log posterior values of the references considered with shape (batch_size,).
    :rtype: torch.Tensor
    :param dev: An object representing the device on which the tensor operations are or will be allocated.
    :type dev: torch.device
    """
    distances_squared = torch.pow(distances, 2)
    log_likelihoods = 0.5 * torch.sum(sigma_lm * (-distances_squared), dim=0)
    log_prior = 0.5 * torch.sum(-torch.pow(parameters / sigma_prior, 2), dim=0)
    uniform_translation_prior = Uniform(torch.tensor(-100, device=dev), torch.tensor(100, device=dev))
    rotation = (rotation + math.pi) % (2.0 * math.pi) - math.pi
    log_translation = torch.sum(uniform_translation_prior.log_prob(translation), dim=0)
    uniform_rotation_prior = Uniform(torch.tensor(-math.pi, device=dev), torch.tensor(math.pi, device=dev))
    log_rotation = torch.sum(uniform_rotation_prior.log_prob(rotation), dim=0)
    return log_likelihoods + log_prior + log_translation + log_rotation


class PDMMetropolisSampler:
    def __init__(self, pdm, proposal, batch_mesh, target, correspondences=True, sigma_lm=1.0, sigma_prior=1.0):
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
        compare them with the target (i.e., calculating the posterior).
        :type batch_mesh: BatchTorchMesh
        :param target: Observed (partial) shape to be analysed.
        :type target: TorchMeshGpu
        :param correspondences: Boolean variable that determines whether there are point correspondences between
        references and target.
        :type correspondences: bool
        :param sigma_lm: Variance used (in square millimetres) when determining the likelihood term of the posterior.
        For the likelihood term, the L2 distance (independent point evaluator likelihood) is used. The L2 distance is
        evaluated using a zero-mean Gaussian with variance sigma_lm.
        :type sigma_lm: float
        :param sigma_prior: Variance used when determining the prior term of the posterior. The prior term pushes the
        solution towards a more likely shape by penalizing unlikely shape deformations.
        :type sigma_prior: float
        """
        self.model = pdm
        self.proposal = proposal
        # Make sure all components are on the same device, i.e., model, proposal, meshes.
        self.dev = self.proposal.dev
        self.batch_mesh = batch_mesh
        self.points = self.batch_mesh.tensor_points
        self.target = target
        self.target_points = target.tensor_points
        self.correspondences = correspondences
        self.batch_size = self.proposal.batch_size
        self.sigma_lm = sigma_lm
        self.sigma_prior = sigma_prior
        self.old_posterior = None
        self.posterior = None
        self.determine_quality(ParameterProposalType.MODEL)
        self.accepted = 0
        self.rejected = 0

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
        if parameter_proposal_type == ParameterProposalType.MODEL:
            reconstructed_points = self.model.get_points_from_parameters(self.proposal.get_parameters())
            if self.batch_size == 1:
                reconstructed_points = reconstructed_points.unsqueeze(2)
            self.batch_mesh.set_points(reconstructed_points)
            self.batch_mesh.apply_translation(self.proposal.get_translation_parameters())
            self.batch_mesh.apply_rotation(self.proposal.get_rotation_parameters())
            self.batch_mesh.old_points = old_points
        elif parameter_proposal_type == ParameterProposalType.TRANSLATION:
            self.batch_mesh.apply_translation(-self.proposal.old_translation)
            self.batch_mesh.apply_translation(self.proposal.get_translation_parameters())
        else:
            self.batch_mesh.apply_rotation(-self.proposal.old_rotation)
            self.batch_mesh.apply_rotation(self.proposal.get_rotation_parameters())
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
        if self.correspondences:
            target_points_expanded = self.target_points.unsqueeze(2).expand(-1, -1, self.proposal.batch_size)
            differences = torch.sub(self.points, target_points_expanded)
            # old draft
            # posterior = unnormalised_posterior(differences, self.proposal.parameters, self.sigma_lm, self.sigma_prior)
            distances = torch.linalg.vector_norm(differences, dim=1)
            posterior = unnormalised_log_posterior(distances, self.proposal.parameters, self.proposal.translation,
                                                   self.proposal.rotation, self.sigma_lm, self.sigma_prior, self.dev)

        else:
            reference_meshes = self.batch_mesh.to_pytorch3d_meshes()
            target_clouds = self.target.to_pytorch3d_pointclouds(self.batch_size)
            # packed representation for faces
            verts_packed = reference_meshes.verts_packed()
            faces_packed = reference_meshes.faces_packed()
            tris = verts_packed[faces_packed]  # (T, 3, 3)
            tris_first_idx = reference_meshes.mesh_to_faces_packed_first_idx()
            # max_tris = reference_meshes.num_faces_per_mesh().max().item()

            points = target_clouds.points_packed()  # (P, 3)
            points_first_idx = target_clouds.cloud_to_packed_first_idx()
            max_points = target_clouds.num_points_per_cloud().max().item()

            # point to face squared distance: shape (P,)
            # requires dtype=torch.float32
            point_to_face_transposed = point_face_distance(points.float(), points_first_idx, tris.float(),
                                                           tris_first_idx, max_points).reshape(
                (-1, int(self.target.num_points)))
            point_to_face = torch.transpose(point_to_face_transposed, 0, 1)

            # Target is a point cloud, reference a mesh.
            posterior = unnormalised_log_posterior(torch.sqrt(point_to_face), self.proposal.parameters,
                                                   self.proposal.translation, self.proposal.rotation, self.sigma_lm,
                                                   self.sigma_prior, self.dev)

        self.posterior = posterior

    def decide(self):
        """
        Implements the second and third step of the Metropolis algorithm (according to the lecture notes of the course
        "Statistical Shape Modelling" by Marcel Lüthi). Calculates the ratio of the unnormalised posterior values of the
        new and old samples. Decides which samples should form the current state based on the rules of the Metropolis
        algorithm. Parameter values (GaussianRandomWalkProposal) and mesh points (BatchMesh) are updated accordingly.

        :return:
        """
        # log-ratio!
        ratio = torch.exp(self.posterior - self.old_posterior)
        probabilities = torch.min(ratio, torch.ones_like(ratio))
        randoms = torch.rand(self.batch_size, device=self.dev)
        decider = torch.gt(probabilities, randoms)
        # decider = torch.ones(self.batch_size).bool()
        self.posterior = torch.where(decider, self.posterior, self.old_posterior)
        self.proposal.update(decider, self.posterior)
        self.batch_mesh.update_points(decider)
        self.points = self.batch_mesh.tensor_points

        self.accepted += decider.sum().item()
        self.rejected += (self.batch_size - decider.sum().item())

    def acceptance_ratio(self):
        """
        Returns the ratio of accepted samples to the total number of random parameter draws.

        :return: Percentage of accepted samples.
        :rtype: float
        """
        return float(self.accepted) / (self.accepted + self.rejected)

    def change_device(self, dev):
        """

        :param dev:
        :return:
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

            self.dev = dev
