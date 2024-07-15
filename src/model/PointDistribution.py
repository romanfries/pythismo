import numpy as np
import torch


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
    inv = torch.inverse(covariance).double()
    normalization = 1.0 / torch.sqrt(torch.pow(torch.tensor(2 * torch.pi), float(k)) * det)
    points_centered = (points - mean).double()
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


def unnormalised_posterior(distances, parameters, sigma_lm, sigma_prior):
    likelihoods = batch_multivariate_gaussian_pdf(3, distances, torch.zeros(3), torch.diag(sigma_lm * torch.ones(3)))
    log_likelihoods = torch.log(likelihoods)
    prior = gaussian_pdf(parameters, sigma=sigma_prior)
    log_prior = torch.log(prior)
    return torch.sum(torch.cat((log_likelihoods, log_prior), dim=0), dim=0)


class PointDistributionModel:
    def __init__(self, meshes):
        self.meshes = meshes
        self.stacked_points = extract_points(self.meshes)
        self.mean = np.mean(self.stacked_points, axis=1)[:, np.newaxis]
        self.points_centered = self.stacked_points - self.mean
        # Avoid explicit representation of the covariance matrix
        # self.covariance = np.cov(self.stacked_points)
        self.sample_size = self.stacked_points.shape[1]
        # Eigenvectors are the columns of the 2-dimensional ndarray 'self.eigenvectors'
        self.eigenvalues, self.eigenvectors = apply_svd(self.points_centered, self.sample_size)
        self.components = self.eigenvectors * np.sqrt(self.eigenvalues)
        # self.parameters = get_parameters(self.points_centered, self.eigenvectors)
        self.parameters = get_parameters(self.points_centered, self.components)

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


class PDMMetropolisSampler:
    def __init__(self, pdm, proposal, batch_mesh, target, correspondences=True, sigma_lm=0.1, sigma_prior=1.0):
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
        self.determine_quality()
        self.accepted = 0
        self.rejected = 0

    def propose(self):
        self.proposal.propose()

    def update_mesh(self):
        reconstructed_points = self.model.get_points_from_parameters(self.proposal.get_parameters().numpy())
        self.batch_mesh.set_points(reconstructed_points, save_old=True)
        self.points = self.batch_mesh.tensor_points

    def determine_quality(self):
        self.update_mesh()
        self.old_posterior = self.posterior
        if self.correspondences:
            target_points_expanded = self.target_points.unsqueeze(2).expand(-1, -1, self.proposal.batch_size)
            distances = torch.sub(self.points, target_points_expanded)
            posterior = unnormalised_posterior(distances, self.proposal.parameters, self.sigma_lm, self.sigma_prior)
        else:
            distances = distance_to_closest_point(self.points, self.target_points, self.batch_size)
            posterior = unnormalised_posterior(distances, self.proposal.parameters, self.sigma_lm, self.sigma_prior)
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






