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


def get_parameters(stacked_points, eigenvectors):
    parameters, residuals, rank, s = np.linalg.lstsq(eigenvectors, stacked_points, rcond=None)
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
    points_centered = points - mean
    exponent = -0.5 * torch.einsum('ijk,jl,ilk->ik', points_centered, inv, points_centered)
    return normalization * torch.exp(exponent)


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
        # self.parameters = get_parameters(self.points_centered, self.eigenvectors)
        self.parameters = get_parameters(self.points_centered, self.eigenvectors)

    def get_eigenvalues(self):
        return self.eigenvalues

    def get_eigenvalue_k(self, k):
        return self.eigenvalues[k]

    def get_eigenvectors(self):
        return self.eigenvectors

    def get_eigenvector_k(self, k):
        return self.eigenvectors[:, k]

    def get_points_from_parameters(self, parameters):
        stacked_points = np.transpose(parameters) @ np.transpose(self.eigenvectors)
        return stacked_points.reshape((-1, 3))


class PDMParameterToMeshConverter:
    def __init__(self, pdm, proposal, batch_mesh, target, correspondences=True, sigma_prior=50.0, sigma_lm=20.0):
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

    def update_mesh(self):
        reconstructed_points = self.model.get_eigenvectors() @ self.proposal.parameters.numpy() + self.model.mean
        reconstructed_points = reconstructed_points.reshape((self.batch_mesh.num_points, self.batch_mesh.dimensionality,
                                                             self.proposal.batch_size))
        self.batch_mesh.set_points(reconstructed_points)
        self.points = self.batch_mesh.tensor_points
        return self.batch_mesh

    def verify(self):
        _ = self.update_mesh()
        if self.correspondences:
            target_points_expanded = self.target_points.unsqueeze(2).expand(-1, -1, self.proposal.batch_size)
            distances = torch.sub(self.points, target_points_expanded)
            zero_mean = torch.zeros(3)
            covariance = torch.diag(self.sigma_lm * torch.ones(3))
            likelihoods = batch_multivariate_gaussian_pdf(3, distances, zero_mean, covariance)
            log_likelihoods = torch.log(likelihoods)
            prior = gaussian_pdf(self.proposal.parameters, sigma=self.sigma_prior)
            log_prior = torch.log(prior)
            posterior_unnormalized = torch.sum(torch.cat((log_likelihoods, log_prior), dim=0), dim=0)

        else:
            # target_points_expanded = self.target_points.unsqueeze(2).expand(-1, -1, self.proposal.batch_size)
            # distances = torch.sum(torch.pow(torch.sub(self.points, target_points_expanded), float(2)), dim=1)

            pass

        return

