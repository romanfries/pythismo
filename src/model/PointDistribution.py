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
    return (np.square(s) / (num_components - 1))[:num_components], np.transpose(V_T)[:, :num_components]


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
    """
    NOTE: Method is currently not used.

    :param ref_points:
    :param target_points:
    :param batch_size:
    :return:
    """
    target_points_expanded = target_points.unsqueeze(2).expand(-1, -1, batch_size).unsqueeze(
        0)
    points_expanded = ref_points.unsqueeze(1)
    distances = torch.sub(points_expanded, target_points_expanded)
    closest_points = torch.argmin(torch.sum(torch.pow(distances, float(2)), dim=2), dim=0)
    distances = distances[closest_points, torch.arange(1000).unsqueeze(1).expand(1000, 100), :,
                torch.arange(100)]
    return torch.transpose(distances, 1, 2)


def index_of_closest_point(ref_points, target_points, batch_size):
    """
    NOTE: Method is currently not used.

    :param ref_points:
    :param target_points:
    :param batch_size:
    :return:
    """
    target_points_expanded = target_points.unsqueeze(2).expand(-1, -1, batch_size).unsqueeze(
        0)
    points_expanded = ref_points.unsqueeze(1)
    distances = torch.sub(points_expanded, target_points_expanded)
    closest_points = torch.argmin(torch.sum(torch.pow(distances, float(2)), dim=2), dim=1)
    return closest_points


def unnormalised_posterior(differences, parameters, sigma_lm, sigma_prior):
    """
    Calculates the unnormalised posterior and then transfers this value to the log space. This method is obsolete and
    ‘unnormalised_log_posterior’ (src/sampling/Metropolis.py) should be used instead, as its calculation is more stable.
    Furthermore, it disregards translation and rotation.
    Further information can be found there.

    :param differences: Tensor with distances of each point of the target to the closest point on the surface of the
    references considered with shape (num_target_points, batch_size).
    :type differences: torch.Tensor
    :param parameters: Tensor with the current model parameters of the references under consideration with shape
    (num_parameters, batch_size).
    :type parameters: torch.Tensor
    :param sigma_lm: Variance of the zero-mean Gaussian, which is used to evaluate the L2 distance. The same variance
    value is used for all 3 dimensions (isotropic distribution).
    :type sigma_lm: float
    :param sigma_prior: Variance of the model parameters (prior term calculation).
    :type sigma_prior: float
    :return: Tensor with unnormalised log posterior values of the references considered with shape (batch_size,).
    :rtype: torch.Tensor
    """
    likelihoods = batch_multivariate_gaussian_pdf(3, differences, torch.zeros(3, dtype=torch.float64),
                                                  torch.diag(sigma_lm * torch.ones(3, dtype=torch.float64)))
    log_likelihoods = torch.log(likelihoods)
    prior = gaussian_pdf(parameters, sigma=sigma_prior)
    log_prior = torch.log(prior)
    return torch.sum(torch.cat((log_likelihoods, log_prior), dim=0), dim=0)


class PointDistributionModel:
    def __init__(self, read_in=False, mean_and_cov=False, meshes=None, mean=None, cov=None, model=None,
                 rank=None):
        """
        Class that defines and creates a point distribution model (PDM).
        This can be done in three different ways: It can be imported from a .h5 file, a complete set of mean vector and
        covariance matrix can be passed, or a batch of meshes can be given from which the PDM is to be calculated.
        Remark: Of the two Boolean values ‘read_in’ and ‘mean_and_cov’, at most one should be True.

        :param read_in: Boolean value that specifies whether a read in model is available.
        :type read_in: bool
        :param mean_and_cov: Boolean value that specifies whether a complete set of mean vector and covariance matrix is
        passed.
        :type mean_and_cov: bool
        :param meshes: List of meshes from which the PDM is to be calculated.
        :type meshes: list
        :param mean: Mean vector with shape (3 * num_points, 1).
        :type mean: np.ndarray
        :param cov: Full covariance matrix with shape (3 * num_points, 3 * num_points).
        :type cov: np.ndarray
        :param model: Read-in PDM in the form of a Python dictionary. To get this dictionary, use
        'src/custom_io/H5ModelIO.py'.
        :type model: dict
        :param rank: Specifies how many main components are to be taken into account. Must only be specified
        explicitly in the case of ‘mean_and_cov=True’.
        :type rank: int
        """
        self.read_in = read_in
        self.mean_and_cov = mean_and_cov
        if self.read_in:
            self.meshes = None
            self.stacked_points = None
            # self.mean = (model.get('points').reshape(-1, order='F') + model.get('mean'))[:, np.newaxis]
            # did not work. Why?
            self.mean = (model.get('points').reshape(-1, order='F'))[:, np.newaxis]
            self.points_centered = None
            self.num_points = int(self.mean.shape[0] / 3)
            self.rank = model.get('basis').shape[1]
            self.eigenvalues = model.get('var')
            self.eigenvectors = model.get('basis')
            self.components = self.eigenvectors * model.get('std')
            self.parameters = None
            self.decimated = False
        elif self.mean_and_cov:
            self.meshes = None
            self.stacked_points = None
            self.mean = mean
            self.points_centered = None
            self.num_points = int(self.mean.shape[0] / 3)
            self.rank = rank
            eigenvalues, self.eigenvectors = apply_svd(cov, self.rank)
            self.eigenvalues = np.sqrt((self.rank - 1) * eigenvalues)
            self.components = self.eigenvectors * np.sqrt(self.eigenvalues)
            self.parameters = None
            self.decimated = False
        else:
            self.meshes = meshes
            self.stacked_points = extract_points(self.meshes)
            self.mean = np.mean(self.stacked_points, axis=1)[:, np.newaxis]
            self.points_centered = self.stacked_points - self.mean
            # Avoid explicit representation of the covariance matrix
            # self.covariance = np.cov(self.stacked_points)
            # Dimensionality of 3 hardcoded here!
            self.num_points = int(self.stacked_points.shape[0] / 3)
            self.rank = self.stacked_points.shape[1]
            # Eigenvectors are the columns of the 2-dimensional ndarray 'self.eigenvectors'
            self.eigenvalues, self.eigenvectors = apply_svd(self.points_centered, self.rank)
            self.components = self.eigenvectors * np.sqrt(self.eigenvalues)
            # self.parameters = get_parameters(self.points_centered, self.eigenvectors)
            self.parameters = get_parameters(self.points_centered, self.components)
            self.decimated = False

    def get_eigenvalues(self):
        """
        Returns all ('rank' many) eigenvalues of the PDM in descending order.

        :return: Numpy array with the eigenvalues in descending order with shape (rank,).
        :rtype: np.ndarray
        """
        return self.eigenvalues

    def get_eigenvalue_k(self, k):
        """
        Returns the kth largest eigenvalue of the PDM.

        :param k: Specifies which eigenvalue is to be returned.
        :type k: int
        :return: The kth largest eigenvalue.
        :rtype: float
        """
        return self.eigenvalues[k]

    def get_components(self):
        """
        Returns all components of the PDM. The kth component of the PDM is defined here as the square root of the kth
        eigenvalue times the kth eigenvector.

        :return: All ('rank' many) components of the PDM with shape (3 * num_points, rank).
        :rtype: np.ndarray
        """
        return self.components

    def get_component_k(self, k):
        """
        Returns the kth component of the PDM. The kth component of the PDM is defined here as the square root of the kth
        eigenvalue times the kth eigenvector.

        :param k: Specifies which component is to be returned.
        :type k: int
        :return: The kth component of the PDM with shape (3 * num_points,).
        :rtype: np.ndarray
        """
        return self.components[:, k]

    def get_points_from_parameters(self, parameters):
        """
        Takes model parameters and calculates the coordinates of all points of the instance defined by these parameters.
        Works for a single set of model parameters as well as for entire batches of model parameters.

        :param parameters: Model parameters for which the point coordinates of the associated instance are to be
        calculated, either of shape (rank,) or (rank, batch_size).
        :type parameters: np.ndarray
        :return: Calculated point coordinates, either of shape (num_points, 3) or (num_points, 3, batch_size).
        :rtype: np.ndarray
        """
        if parameters.ndim == 1:
            parameters = parameters[:, np.newaxis]
        batch_size = parameters.shape[1]
        stacked_points = self.components @ parameters + self.mean
        if batch_size == 1:
            return stacked_points.reshape((-1, 3))
        else:
            return stacked_points.reshape((-1, 3, batch_size))

    def decimate(self, decimation_target=200):
        """
        Reduces the PDM to the specified number of points. Currently, only calculated PDMs can be reduced, i.e. if
        ‘mean_and_cov'='read_in'=False.

        :param decimation_target: Number of target points that the reduced PDM should have.
        :type decimation_target: int
        :return: The first mesh is reduced to the number of target points using QEM, then the PDM is defined at these
        points using interpolation (radial basis functions). This mesh instance is returned.
        :rtype: TorchMesh
        """
        reference_decimated = self.meshes[0]
        if self.read_in or self.mean_and_cov:
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
        mean_fun, cov_fun = self.pdm_to_interpolated_gp(decimation_target)
        mean, cov = mean_fun(reference_decimated.points), cov_fun(reference_decimated.points,
                                                                       reference_decimated.points)
        self.mean = mean.reshape((-1, 1))
        eigenvalues, eigenvectors = apply_svd(cov, self.rank)
        self.eigenvalues = np.sqrt(eigenvalues[:self.rank] * (self.rank - 1))
        self.eigenvectors = eigenvectors[:, :self.rank]
        self.meshes = None
        self.stacked_points = None
        self.points_centered = None
        self.num_points = reference_decimated.num_points
        self.components = self.eigenvectors * np.sqrt(self.eigenvalues)
        self.parameters = None
        self.decimated = True

        return reference_decimated

    def pdm_to_interpolated_gp(self, decimation_target):
        """
        Extends the PDM, which is only defined at a finite number of points, to an infinite domain. This then
        corresponds to a Gaussian process (GP). The values of the deformation field between the defined points are
        interpolated using a radial basis functions (RBF) interpolator.

        :param decimation_target: Number of target points that the reduced PDM should have.
        :type decimation_target: int
        :return: A tuple containing two functions.
            - The first function is the mean function of the GP.
            - The second function is the covariance function of the GP.
        :rtype: Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray, np.ndarray], np.ndarray]]
        """
        mean_reshaped = self.mean.reshape((-1, 3))
        eigenvectors_reshaped = self.eigenvectors.reshape((self.num_points, 3, self.rank))
        eigenvector_interpolators = [RBFInterpolator(mean_reshaped, eigenvectors_reshaped[:, :, i]) for i in
                                     range(self.rank)]
        def mean(x):
            """
            Evaluates the mean function at a given set of points x.

            :param x: Points at which the covariance function is to be analysed with shape (num_points, 3).
            :type x: np.ndarray
            :return: Mean function evaluated at all the points in x with shape (num_points, 3).
            :rtype: np.ndarray
            """
            return RBFInterpolator(mean_reshaped, mean_reshaped)(x)

        def cov(x, y, num_points=decimation_target):
            """
            Evaluates the covariance function at given set of points x, y. The number of points in x and y must be
            identical.

            :param x: Points at which the covariance function is to be analysed with shape (num_points, 3).
            :type x: np.ndarray
            :param y: Points at which the covariance function is to be analysed with shape (num_points, 3).
            :type y: np.ndarray
            :param num_points: Indicates how many points are present in the sets x, y.
            :type num_points: int
            :return: Full covariance matrix that determines the covariance for each point in x with each point from y
            [shape (3 * num_points, 3 * num_points)].
            :rtype: np.ndarray
            """
            x_eigen = np.dstack([interp(x) for interp in eigenvector_interpolators]).reshape((3 * num_points, self.rank))
            y_eigen = np.dstack([interp(y) for interp in eigenvector_interpolators]).reshape((3 * num_points, self.rank))

            return (x_eigen * self.eigenvalues) @ np.transpose(y_eigen)

        return mean, cov

    def get_covariance(self):
        """
        Calculates and returns the full covariance matrix of the PDM. For reasons of storage efficiency, the full
        covariance matrix is not permanently stored.

        :return: Full covariance matrix with shape (3 * num_points, 3 * num_points).
        :rtype: np.ndarray
        """
        if not self.read_in and not self.mean_and_cov and not self.decimated:
            return np.cov(self.stacked_points)
        else:
            return self.eigenvectors @ (np.diag(self.eigenvalues) @ np.transpose(self.eigenvectors))





