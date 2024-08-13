import numpy as np
import pytorch3d
import torch
import trimesh.base
from pytorch3d.loss.point_mesh_distance import point_face_distance
from trimesh import Trimesh

from src.mesh.TMesh import BatchTorchMesh
from src.model.PointDistribution import PointDistributionModel
from src.sampling.proposals.GaussRandWalk import ParameterProposalType, GaussianRandomWalkProposal


class ClosestPointProposal(GaussianRandomWalkProposal):

    def __init__(self, batch_size, starting_parameters, reference, target, model, sigma_mod=0.005, sigma_trans=0.1,
                 sigma_rot=0.0001, chain_length_step=1000):
        super().__init__(batch_size, starting_parameters, sigma_mod, sigma_trans, sigma_rot, chain_length_step)
        self.reference = reference.copy()
        self.target = target
        # self.target = BatchTorchMesh(target, target.id, batch_size=1)
        self.prior_model = model

    def calculate_posterior_model(self, m=100, sigma_n=3.0, sigma_v=100):
        generator = np.random.default_rng()
        rint = generator.integers(0, self.batch_size, 1)
        reconstructed_points = self.prior_model.get_points_from_parameters(self.get_parameters().numpy()[:, rint])
        self.reference.set_points(reconstructed_points)

        # 1: Sample m points {s_i} on the current model instance \Gamma[\alpha].
        sampled_indexes = torch.tensor(np.random.choice(np.arange(0, self.prior_model.num_points), m, replace=False), dtype=torch.int64)

        # 2: For every point s_i, i \in [0, ..., m] find the closest point c_i on the target \Gamma_T.
        target_mesh = Trimesh(self.target.tensor_points, torch.tensor(self.target.cells[0].data))
        sampled_points = torch.index_select(self.reference.tensor_points, 0, sampled_indexes)
        closest_points, _, _ = target_mesh.nearest.on_surface(sampled_points)
        closest_points = torch.tensor(closest_points)

        # 3: Construct the set of observations L based on corresponding landmark pairs (s_i, c_i) according to eq. 9
        # and define the noise \epsilon_i \distas \mathcal{N}(0, \Sigma_{s_{i}} using eq. 16
        differences = closest_points - sampled_points
        landmark_set = torch.stack((sampled_points, differences), dim=2)
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
        mean = torch.tensor(self.prior_model.mean, dtype=torch.float64)
        cov = torch.tensor(self.prior_model.get_covariance(), dtype=torch.float64)

        sampled_indexes_3d = torch.cat([sampled_indexes * 3 + i for i in range(3)], dim=0)
        sampled_covariances = covariances[sampled_indexes, :, :]
        K_yy = cov[sampled_indexes_3d][:, sampled_indexes_3d]
        covariances_reshaped = torch.zeros((3 * m, 3 * m), dtype=torch.float64)
        covariances_reshaped[torch.arange(3 * m).repeat_interleave(3), torch.arange(3 * m).view(-1, 3).repeat_interleave(3, dim=0).flatten()] = sampled_covariances.flatten()
        K_yy_inv = torch.inverse(K_yy + covariances_reshaped)
        K_xy = cov[:, sampled_indexes_3d]
        posterior_mean = mean + K_xy @ K_yy_inv @ (landmark_set[:,:,1].transpose(0, 1).flatten().unsqueeze(1) - mean[sampled_indexes_3d])
        posterior_cov = cov - K_xy @ K_yy_inv @ K_xy.transpose(0, 1)

        self.posterior_model = PointDistributionModel(mean_and_cov=True, mean=posterior_mean.numpy(), cov=posterior_cov.numpy(), sample_size=self.prior_model.sample_size)

        return self.posterior_model






