import math
import warnings

import numpy as np
import torch

from pytorch3d.loss.point_mesh_distance import point_face_distance, face_point_distance
from torch.distributions import Uniform

from src.sampling.proposals.GaussRandWalk import ParameterProposalType


def extract_points(meshes):
    for index, mesh in enumerate(meshes):
        if index == 0:
            stacked_points = mesh.points.ravel()
        else:
            stacked_points = np.column_stack((stacked_points, mesh.points.ravel()))
    return stacked_points


def apply_svd(centered_points, num_components):
    _, s, V_T = np.linalg.svd(np.transpose(centered_points), full_matrices=False)
    # The rows of V_T are the eigenvector of the covariance matrix. The singular values are related to the eigenvalues
    # of the covariance matrix via $\lambda_i = s_i^2/(n-1)$. For further information see
    # https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
    return np.square(s) / (num_components - 1), np.transpose(V_T)


def get_parameters(stacked_points, components):
    parameters, residuals, rank, s = np.linalg.lstsq(components, stacked_points, rcond=None)
    return parameters


def gaussian_pdf(x, mean=0.0, sigma=1.0):
    normalization = 1.0 / (sigma * torch.sqrt(torch.tensor(2.0 * torch.pi)))
    exponent = torch.exp(-0.5 * ((x - mean) / sigma) ** 2)
    return normalization * exponent


def batch_multivariate_gaussian_pdf(k, points, mean, covariance):
    # This function is tailored to tensors of shape (num_points, dimensionality, batch_size) to calculate the likelihood
    # for every point of a batch mesh.
    mean = mean.unsqueeze(0).unsqueeze(2)
    det = torch.det(covariance)
    inv = torch.inverse(covariance)
    normalization = 1.0 / torch.sqrt(torch.pow(torch.tensor(2 * torch.pi), float(k)) * det)
    points_centered = (points - mean)
    exponent = -0.5 * torch.einsum('ijk,jl,ilk->ik', points_centered, inv, points_centered)
    return normalization * torch.exp(exponent)


def distance_to_closest_point(ref_points, target_points, batch_size):
    target_points_expanded = target_points.unsqueeze(2).expand(-1, -1, batch_size).unsqueeze(
        0)
    points_expanded = ref_points.unsqueeze(1)
    distances = torch.sub(points_expanded, target_points_expanded)
    closest_points = torch.argmin(torch.sum(torch.pow(distances, float(2)), dim=2), dim=0)
    distances = distances[closest_points, torch.arange(1000).unsqueeze(1).expand(1000, 100), :,
                torch.arange(100)]
    return torch.transpose(distances, 1, 2)


def index_of_closest_point(ref_points, target_points, batch_size):
    target_points_expanded = target_points.unsqueeze(2).expand(-1, -1, batch_size).unsqueeze(
        0)
    points_expanded = ref_points.unsqueeze(1)
    distances = torch.sub(points_expanded, target_points_expanded)
    closest_points = torch.argmin(torch.sum(torch.pow(distances, float(2)), dim=2), dim=1)
    return closest_points


def unnormalised_posterior(differences, parameters, sigma_lm, sigma_prior):
    likelihoods = batch_multivariate_gaussian_pdf(3, differences, torch.zeros(3, dtype=torch.float64),
                                                  torch.diag(sigma_lm * torch.ones(3, dtype=torch.float64)))
    log_likelihoods = torch.log(likelihoods)
    prior = gaussian_pdf(parameters, sigma=sigma_prior)
    log_prior = torch.log(prior)
    return torch.sum(torch.cat((log_likelihoods, log_prior), dim=0), dim=0)


def unnormalised_log_posterior(distances, parameters, translation, rotation, sigma_lm, sigma_prior):
    distances_squared = torch.pow(distances, 2)
    log_likelihoods = 0.5 * torch.sum(sigma_lm * (-distances_squared), dim=0)
    log_prior = 0.5 * torch.sum(-torch.pow(parameters / sigma_prior, 2), dim=0)
    uniform_translation_prior = Uniform(-100, 100)
    rotation = (rotation + math.pi) % (2.0 * math.pi) - math.pi
    log_translation = torch.sum(uniform_translation_prior.log_prob(translation), dim=0)
    uniform_rotation_prior = Uniform(-math.pi, math.pi)
    log_rotation = torch.sum(uniform_rotation_prior.log_prob(rotation), dim=0)
    return log_likelihoods + log_prior + log_translation + log_rotation


class PointDistributionModel:
    def __init__(self, meshes=None, read_in=False, model=None):
        if not read_in:
            self.meshes = meshes
            self.stacked_points = extract_points(self.meshes)
            self.mean = np.mean(self.stacked_points, axis=1)[:, np.newaxis]
            self.points_centered = self.stacked_points - self.mean
            # Avoid explicit representation of the covariance matrix
            # self.covariance = np.cov(self.stacked_points)
            # Dimensionality of 3 hardcoded here!
            self.num_points = self.stacked_points.shape[0] / 3
            self.sample_size = self.stacked_points.shape[1]
            # Eigenvectors are the columns of the 2-dimensional ndarray 'self.eigenvectors'
            self.eigenvalues, self.eigenvectors = apply_svd(self.points_centered, self.sample_size)
            self.components = self.eigenvectors * np.sqrt(self.eigenvalues)
            # self.parameters = get_parameters(self.points_centered, self.eigenvectors)
            self.parameters = get_parameters(self.points_centered, self.components)
            self.decimated = False
        else:
            self.meshes = None
            self.stacked_points = None
            # self.mean = (model.get('points').reshape(-1, order='F') + model.get('mean'))[:, np.newaxis]
            # did not work. Why?
            self.mean = (model.get('points').reshape(-1, order='F'))[:, np.newaxis]
            self.points_centered = None
            self.num_points = self.mean.shape[0] / 3
            self.sample_size = model.get('basis').shape[1]
            self.eigenvalues = model.get('var')
            self.eigenvectors = model.get('basis')
            self.components = self.eigenvectors * model.get('std')
            self.parameters = None

    def get_eigenvalues(self):
        return self.eigenvalues

    def get_eigenvalue_k(self, k):
        return self.eigenvalues[k]

    def get_components(self):
        return self.components

    def get_component_k(self, k):
        return self.components[:, k]

    def get_points_from_parameters(self, parameters):
        if parameters.ndim == 1:
            parameters = parameters[:, np.newaxis]
        batch_size = parameters.shape[1]
        stacked_points = self.components @ parameters + self.mean
        if batch_size == 1:
            return stacked_points.reshape((-1, 3))
        else:
            return stacked_points.reshape((-1, 3, batch_size))

    def decimate(self, decimation_target=200):

        reference_decimated = self.meshes[0]
        if self.meshes is None:
            warnings.warn("Warning: Decimation of imported Point Distribution Models is not (yet) supported.",
                          UserWarning)
            return reference_decimated

        if self.num_points <= decimation_target:
            warnings.warn("Warning: No decimation necessary, as the target is below the current number of points.",
                          UserWarning)
            return reference_decimated

        if self.decimated:
            warnings.warn("Warning: Point Distribution Model can only be decimated once.",
                          UserWarning)
            return reference_decimated

        reference_decimated = self.meshes[0].simplify_qem(decimation_target)
        closest_points = index_of_closest_point(reference_decimated.tensor_points.unsqueeze(2),
                                                torch.tensor(self.mean.reshape(-1, 3)), batch_size=1)
        closest_points_flat = closest_points.flatten() * 3
        indices = np.vstack((closest_points_flat, closest_points_flat + 1, closest_points_flat + 2)).T.flatten()
        self.meshes = None
        self.stacked_points = None
        self.points_centered = None
        self.num_points = reference_decimated.num_points
        self.mean = self.mean[indices]
        self.eigenvectors = self.eigenvectors[indices, :]
        self.components = self.components[indices, :]
        self.parameters = None
        self.decimated = True

        return reference_decimated


class PDMMetropolisSampler:
    def __init__(self, pdm, proposal, batch_mesh, target, correspondences=True, sigma_lm=1.0, sigma_prior=1.0):
        self.model = pdm
        self.proposal = proposal
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
        self.proposal.propose(parameter_proposal_type)

    def update_mesh(self, parameter_proposal_type: ParameterProposalType):
        if parameter_proposal_type == ParameterProposalType.MODEL:
            reconstructed_points = self.model.get_points_from_parameters(self.proposal.get_parameters().numpy())
            if self.batch_size == 1:
                reconstructed_points = reconstructed_points[:, :, np.newaxis]
            self.batch_mesh.set_points(reconstructed_points, save_old=True)
        elif parameter_proposal_type == ParameterProposalType.TRANSLATION:
            self.batch_mesh.apply_translation(self.proposal.get_translation_parameters().numpy(), save_old=True)
        else:
            self.batch_mesh.apply_rotation(self.proposal.get_rotation_parameters().numpy(), save_old=True)
        self.points = self.batch_mesh.tensor_points

    def determine_quality(self, parameter_proposal_type: ParameterProposalType):
        self.update_mesh(parameter_proposal_type)
        self.old_posterior = self.posterior
        if self.correspondences:
            target_points_expanded = self.target_points.unsqueeze(2).expand(-1, -1, self.proposal.batch_size)
            differences = torch.sub(self.points, target_points_expanded)
            # old draft
            # posterior = unnormalised_posterior(differences, self.proposal.parameters, self.sigma_lm, self.sigma_prior)
            distances = torch.linalg.vector_norm(differences, dim=1)
            posterior = unnormalised_log_posterior(distances, self.proposal.parameters, self.proposal.translation, self.proposal.rotation, self.sigma_lm, self.sigma_prior)

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
                                                           tris_first_idx, max_points).double().reshape(
                (-1, int(self.model.num_points)))
            point_to_face = torch.transpose(point_to_face_transposed, 0, 1)

            # Target is a point cloud, reference a mesh.
            posterior = unnormalised_log_posterior(torch.sqrt(point_to_face), self.proposal.parameters, self.proposal.translation, self.proposal.rotation, self.sigma_lm,
                                                   self.sigma_prior)

        self.posterior = posterior

    def decide(self):
        # log-ratio!
        ratio = torch.exp(self.posterior - self.old_posterior)
        probabilities = torch.min(ratio, torch.ones_like(ratio))
        randoms = torch.rand(self.batch_size)
        decider = torch.gt(probabilities, randoms)
        # decider = torch.ones(self.batch_size).bool()

        self.proposal.update(decider)
        self.batch_mesh.update_points(decider)
        self.points = self.batch_mesh.tensor_points

        self.accepted += decider.sum().item()
        self.rejected += (self.batch_size - decider.sum().item())

    def acceptance_ratio(self):
        return float(self.accepted) / (self.accepted + self.rejected)
