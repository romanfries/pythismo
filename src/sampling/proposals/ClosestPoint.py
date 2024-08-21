import numpy as np
import torch
import trimesh.base
from trimesh import Trimesh

from src.mesh.TMesh import BatchTorchMesh
from src.model.PointDistribution import PointDistributionModel, get_parameters
from src.sampling.proposals.GaussRandWalk import ParameterProposalType, GaussianRandomWalkProposal


class ClosestPointProposal(GaussianRandomWalkProposal):

    def __init__(self, batch_size, starting_parameters, reference, batched_reference, target, model, sigma_mod=0.1,
                 sigma_trans=0.1, sigma_rot=0.0001, chain_length_step=1000):
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
        this case, the current state equals the starting state, as the posterior model is only calculated at the
        beginning.)
        All types of parameters (posterior model and pose) are drawn independently of a Gaussian distribution with mean
        value at the previous parameter value and a standardised variance. The variance is defined separately for all 3
        types of parameters.

        :param batch_size: Number of proposals to be generated simultaneously.
        :type batch_size: int
        :param starting_parameters: Start values of the model parameters. When new parameters are drawn for the first
        time, these initial values are used as the mean value of the Gaussian distribution (for all the elements of the
        batch). The shape of the array is assumed to be (num_model_parameters,).
        :type starting_parameters: np.ndarray
        :param reference: Single reference mesh. Necessary for calculating the analytic posterior.
        :type reference: TorchMesh
        :param batched_reference: Current batch with reference meshes. When the analytic posterior is calculated, an
        element of the batch is randomly selected from whose points the distances to the target are determined.
        (Redundant at the moment, but at the beginning the class was designed so that the analytic posterior is
        recalculated after each iteration.)
        :type batched_reference: BatchTorchMesh
        :param target: Instance to be fitted. Necessary for calculating the analytic posterior.
        :type target: TorchMesh
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
        super().__init__(batch_size, starting_parameters, sigma_mod, sigma_trans, sigma_rot, chain_length_step)
        self.posterior_parameters = torch.tensor(
            np.tile(np.zeros(model.sample_size)[:, np.newaxis], (1, self.batch_size)))
        self.reference = reference.copy()
        self.target = target
        # self.target = BatchTorchMesh(target, target.id, batch_size=1)
        self.prior_model = model
        self.posterior_model = self.calculate_posterior_model(batched_reference)
        self.old_posterior_parameters = None

    def calculate_posterior_model(self, batched_reference, sigma_n=3.0, sigma_v=100):
        """
        Executes the calculation of the posterior model.
        In contrast to the algorithm in the paper by D. Madsen et al., all points on the selected current model instance
        are used as observations.
        The step-length mentioned in step 6 is set to 1.0 (currently without the possibility of adjustment). With a step
        size of 1.0, the proposed sample is an independent sample from the calculated posterior.
        The noise of the observations is modeled with low variance along the normal direction and high variance along
        the surface. For this reason, the corresponding facet and vertex normals must be calculated along the way.

        :param batched_reference: Current batch with reference meshes. When the analytic posterior is calculated, an
        element of the batch is randomly selected from whose points the distances to the target are determined.
        (Redundant at the moment, but at the beginning the class was designed so that the analytic posterior is
        recalculated after each iteration.)
        :type batched_reference: BatchTorchMesh
        :param sigma_n: Variance along the normal direction.
        :type sigma_n: float
        :param sigma_v: Variance along the surface.
        :type sigma_v: float
        :return: Calculated posterior model.
        :rtype: PointDistributionModel
        """
        # TODO: Rewrite method so that partial targets are supported. To do this, the distances from all points of the
        #   target to the next point on the chosen reference mesh must be considered (?).
        # The method still contains the original code, which attempted to select only a few points as observations
        # (consistent with Madsen's algorithm).
        self.posterior_parameters = torch.tensor(
            np.tile(np.zeros(self.prior_model.sample_size)[:, np.newaxis], (1, self.batch_size)))
        m = self.prior_model.num_points
        generator = np.random.default_rng()
        rint = generator.integers(0, self.batch_size, 1)
        reconstructed_points = np.squeeze(batched_reference.points[:, :, rint])
        # reconstructed_points = self.prior_model.get_points_from_parameters(self.get_parameters().numpy()[:, rint])
        self.reference.set_points(reconstructed_points)

        # 1: Sample m points {s_i} on the current model instance \Gamma[\alpha].
        # sampled_indexes = torch.tensor(np.random.choice(np.arange(0, self.prior_model.num_points), m, replace=False),
        #                               dtype=torch.int64)

        # 2: For every point s_i, i \in [0, ..., m] find the closest point c_i on the target \Gamma_T.
        target_mesh = Trimesh(self.target.tensor_points, torch.tensor(self.target.cells[0].data))
        # sampled_points = torch.index_select(self.reference.tensor_points, 0, sampled_indexes)
        # closest_points, _, _ = target_mesh.nearest.on_surface(sampled_points)
        closest_points, _, _ = target_mesh.nearest.on_surface(self.reference.tensor_points)
        closest_points = torch.tensor(closest_points)

        # 3: Construct the set of observations L based on corresponding landmark pairs (s_i, c_i) according to eq. 9
        # and define the noise \epsilon_i \distas \mathcal{N}(0, \Sigma_{s_{i}} using eq. 16
        # differences = closest_points - sampled_points
        differences = closest_points - self.reference.tensor_points
        # landmark_set = torch.stack((sampled_points, differences), dim=2)
        landmark_set = torch.stack((self.reference.tensor_points, differences))
        self.reference.calc_facet_normals()
        mean_vertex_normals = torch.tensor(trimesh.geometry.mean_vertex_normals(self.prior_model.num_points,
                                                                                torch.tensor(
                                                                                    self.reference.cells[0].data),
                                                                                self.reference.cell_data[
                                                                                    "facet_normals"]))
        mean_vertex_normals = mean_vertex_normals / torch.norm(mean_vertex_normals, dim=1, keepdim=True)

        # Calculate 2 perpendicular vectors for each normal.
        # Select a reference vector that is not parallel to most of the normals.
        ref = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)

        # To avoid issues where the normal might be parallel to the reference vector
        # we create a matrix to select the next-best vector for those cases.
        parallel_mask = (torch.abs(mean_vertex_normals @ ref) > 0.9).unsqueeze(1)
        alt = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        ref = torch.where(parallel_mask, alt, ref)

        # Compute the first tangent vector by taking the cross product of the normal and the reference vector.
        v1 = torch.cross(mean_vertex_normals, ref.expand_as(mean_vertex_normals), dim=1)
        v1 = v1 / torch.norm(v1, dim=1, keepdim=True)

        # Compute the second tangent vector by taking the cross product of the normal and the first tangent.
        v2 = torch.cross(mean_vertex_normals, v1, dim=1)
        v2 = v2 / torch.norm(v2, dim=1, keepdim=True)

        directions = torch.stack((mean_vertex_normals, v1, v2), dim=2)
        sigma = torch.diag(torch.tensor([sigma_n, sigma_v, sigma_v], dtype=torch.float64))
        covariances = directions @ sigma @ directions.transpose(1, 2)

        # 4: Compute the analytic posterior \mathcal{M}_{\alpha} (eq. 10) with L and {\Sigma_{s_{i}}}.
        mean_prior = torch.tensor(self.prior_model.mean, dtype=torch.float64)
        cov_prior = torch.tensor(self.prior_model.get_covariance(), dtype=torch.float64)

        # sampled_indexes_3d = torch.cat([sampled_indexes * 3 + i for i in range(3)], dim=0)
        # sampled_covariances = covariances[sampled_indexes, :, :]
        # K_yy = cov_prior[sampled_indexes_3d][:, sampled_indexes_3d]
        cov_reshaped = torch.zeros((3 * m, 3 * m), dtype=torch.float64)
        indices = torch.arange(0, 3 * m, 3).unsqueeze(1) + torch.arange(3).unsqueeze(0)
        # cov_reshaped[torch.arange(3 * m).repeat_interleave(3),
        #             torch.arange(3 * m).view(-1, 3).repeat_interleave(3, dim=0).flatten()] = sampled_covariances \
        #    .flatten()
        cov_reshaped[indices.unsqueeze(-1), indices.unsqueeze(-2)] += covariances

        K_yy_inv = torch.inverse(cov_prior + cov_reshaped)
        # K_xy = cov_prior[:, sampled_indexes_3d]
        # posterior_mean = mean_prior + K_xy @ K_yy_inv @ (
        #            landmark_set[:, :, 1].transpose(0, 1).flatten().unsqueeze(1) - mean_prior[sampled_indexes_3d])
        # posterior_cov = cov_prior - K_xy @ K_yy_inv @ K_xy.transpose(0, 1)
        # mean_posterior = mean_prior + cov_prior @ K_yy_inv @ (
        #            torch.tensor(reconstructed_points.flatten()).unsqueeze(-1) - mean_prior)
        cov_posterior = cov_prior - cov_prior @ K_yy_inv @ cov_prior
        self.posterior_model = PointDistributionModel(mean_and_cov=True, mean=mean_prior.numpy(),
                                                      cov=cov_posterior.numpy(),
                                                      sample_size=self.prior_model.sample_size)
        return self.posterior_model

    def propose(self, parameter_proposal_type: ParameterProposalType):
        """
        Updates the parameter values using a Gaussian random walk in posterior space (see class description).

        :param parameter_proposal_type: Specifies which parameters are to be drawn.
        :type parameter_proposal_type: ParameterProposalType
        """
        if parameter_proposal_type == ParameterProposalType.MODEL:
            perturbations = torch.randn((self.num_parameters, self.batch_size))
        else:
            perturbations = torch.randn((3, self.batch_size))

        self.old_parameters = self.parameters
        self.old_posterior_parameters = self.posterior_parameters
        self.old_translation = self.translation
        self.old_rotation = self.rotation

        if parameter_proposal_type == ParameterProposalType.MODEL:
            self.posterior_parameters = self.posterior_parameters + perturbations * self.sigma_mod
            reconstructed_points = self.posterior_model.get_points_from_parameters(self.posterior_parameters.numpy())
            # self.parameters = torch.tensor(np.transpose(self.prior_model.components) @ (
            #            reconstructed_points.reshape((3 * self.prior_model.num_points, -1)) - self.prior_model.mean))
            self.parameters = torch.tensor(get_parameters(
                reconstructed_points.reshape((3 * self.prior_model.num_points, -1)) - self.prior_model.mean,
                self.prior_model.components))
        elif parameter_proposal_type == ParameterProposalType.TRANSLATION:
            self.translation = self.translation + perturbations * self.sigma_trans
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
