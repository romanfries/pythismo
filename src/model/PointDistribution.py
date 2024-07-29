import warnings

import numpy as np
import torch

from scipy.interpolate import RBFInterpolator


def extract_points(meshes):
    """
    Creates a large data matrix with all mesh points from a list of meshio.Mesh objects.

    :param meshes: List with the meshes. It is assumed that these meshes are in correspondence.
    :type meshes: list of meshio.Mesh
    :return: Data matrix with shape (3 * num_points, num_meshes).
    :rtype: np.ndarray
    """
    for index, mesh in enumerate(meshes):
        if index == 0:
            stacked_points = mesh.points.ravel()
        else:
            stacked_points = np.column_stack((stacked_points, mesh.points.ravel()))
    return stacked_points


def apply_svd(centered_points, num_components):
    """
    Applies SVD (singular value decomposition) to a centred data matrix and returns the first ‘num_components’
    eigenvector / eigenvalue pairs to create a point distribution model (PDM). For more information on the relationship
    between PCA and SVD see:
    https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca

    :param centered_points: Centred data matrix with shape (num_points, num_samples).
    :type centered_points: np.ndarray
    :param num_components: Indicates how many eigenvector / eigenvalue are considered (rank of the model).
    :type num_components: int
    :return: Tuple with two elements:
        - np.ndarray: The first ‘num_components’ eigenvalues in descending order with shape (num_components,).
        - np.ndarray: The corresponding first ‘num_components’ eigenvectors with shape (num_points, num_components).
    """
    _, s, V_T = np.linalg.svd(np.transpose(centered_points), full_matrices=False)
    # The rows of V_T are the eigenvector of the covariance matrix. The singular values are related to the eigenvalues
    # of the covariance matrix via $\lambda_i = s_i^2/(n-1)$.
    return np.square(s) / (num_components - 1), np.transpose(V_T)


def get_parameters(stacked_points, components):
    """
    Determines the model parameters in the model space that correspond to the given mesh points. This involves solving
    a least squares problem.

    :param stacked_points: Batch with given mesh points with shape (3 * num_points, batch_size).
    :type stacked_points: np.ndarray
    :param components: Eigenvectors of the model multiplied by the square root of the respective eigenvalue with shape
    (3 * num_points, model_rank).
    :type components: np.ndarray
    :return: Calculated model parameters with shape (model_rank, batch_size).
    :rtype: np.ndarray
    """
    parameters, residuals, rank, s = np.linalg.lstsq(components, stacked_points, rcond=None)
    return parameters


def gaussian_pdf(x, mean=0.0, sigma=1.0):
    """
    Calculates the likelihood of points in 1-dimensional space assuming a Gaussian distribution, defined by
    the input parameters ‘mean’ and ‘covariance’. Supports batched input points.

    :param x: Tensor of points whose likelihoods are to be calculated with shape (num_points, batch_size).
    :type x: torch.Tensor
    :param mean: Mean value of the normal distribution.
    :type mean: float
    :param sigma: Variance of the normal distribution.
    :type sigma: float
    :return: Tensor with likelihoods of the input points assuming the given parameters with the same shape as 'x', i.e.,
    (num_points, batch_size).
    :rtype: torch.Tensor
    """
    normalization = 1.0 / (sigma * torch.sqrt(torch.tensor(2.0 * torch.pi)))
    exponent = torch.exp(-0.5 * ((x - mean) / sigma) ** 2)
    return normalization * exponent


def batch_multivariate_gaussian_pdf(k, points, mean, covariance):
    """
    Calculates the likelihood of points in k-dimensional space assuming a multivariate Gaussian distribution, defined by
    the input parameters ‘mean’ and ‘covariance’. Supports batched input points.

    :param k: Dimensionality of the space.
    :type k: int
    :param points: Tensor of points whose likelihoods are to be calculated with shape (num_points, k, batch_size).
    :type points: torch.Tensor
    :param mean: Tensor with mean value in each dimension with shape (k,).
    :type mean: torch.Tensor
    :param covariance: Covariance matrix tensor with shape (k, k).
    :type covariance: torch.Tensor
    :return: Tensor with likelihoods of the input points assuming the given parameters with shape
    (num_points, batch_size).
    :rtype: torch.Tensor
    """
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
            self.num_points = int(self.stacked_points.shape[0] / 3)
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
            self.num_points = int(self.mean.shape[0] / 3)
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
        mean_fun, cov_fun = self.pdm_to_low_rank_gp(decimation_target)
        mean, cov = mean_fun(reference_decimated.points), cov_fun(reference_decimated.points,
                                                                       reference_decimated.points)
        self.mean = mean.reshape((-1, 1))
        eigenvalues, eigenvectors = apply_svd(cov, self.sample_size)
        self.eigenvalues = np.sqrt(eigenvalues[:self.sample_size] * (self.sample_size - 1))
        self.eigenvectors = eigenvectors[:, :self.sample_size]
        self.meshes = None
        self.stacked_points = None
        self.points_centered = None
        self.num_points = reference_decimated.num_points
        self.components = self.eigenvectors * np.sqrt(self.eigenvalues)
        self.parameters = None
        self.decimated = True

        return reference_decimated

    def pdm_to_low_rank_gp(self, decimation_target):
        mean_reshaped = self.mean.reshape((-1, 3))
        eigenvectors_reshaped = self.eigenvectors.reshape((self.num_points, 3, self.sample_size))
        eigenvector_interpolators = [RBFInterpolator(mean_reshaped, eigenvectors_reshaped[:, :, i]) for i in
                                     range(self.sample_size)]
        def mean(x):
            return RBFInterpolator(mean_reshaped, mean_reshaped)(x)

        def cov(x, y):
            x_eigen = np.dstack([interp(x) for interp in eigenvector_interpolators]).reshape((3 * decimation_target, self.sample_size))
            y_eigen = np.dstack([interp(y) for interp in eigenvector_interpolators]).reshape((3 * decimation_target, self.sample_size))

            return (x_eigen * self.eigenvalues) @ np.transpose(y_eigen)

        return mean, cov






