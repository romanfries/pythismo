import warnings
import numpy as np
from scipy.interpolate import RBFInterpolator

import torch


def extract_points(meshes):
    """
    Creates a large data matrix with all mesh points from a list of meshio.Mesh objects.

    :param meshes: List with the meshes. It is assumed that these meshes are in correspondence.
    :type meshes: list of TorchMeshGpu
    :return: Data matrix with shape (3 * num_points, num_meshes).
    :rtype: torch.Tensor
    """
    stacked_points = torch.tensor([], device=meshes[0].dev)
    for index, mesh in enumerate(meshes):
        if index == 0:
            stacked_points = mesh.tensor_points.flatten()
        else:
            stacked_points = torch.column_stack((stacked_points, mesh.tensor_points.flatten()))

    return stacked_points


def apply_svd(centered_points, num_components):
    """
    Applies SVD (singular value decomposition) to a centered data matrix and returns the first 'num_components'
    eigenvector/eigenvalue pairs to create a point distribution model (PDM). For more information on the relationship
    between PCA and SVD see:
    https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca

    :param centered_points: Centered data matrix with shape (num_points, num_samples).
    :type centered_points: torch.Tensor
    :param num_components: Indicates how many eigenvector/eigenvalue pairs are considered (rank of the model).
    :type num_components: int
    :return: Tuple with two elements:
        - torch.Tensor: The first 'num_components' eigenvalues in descending order with shape (num_components,).
        - torch.Tensor: The corresponding first 'num_components' eigenvectors with shape (num_points, num_components).
    """
    # Perform SVD using PyTorch
    U, s, V_T = torch.svd(centered_points.t(), some=True)
    eigenvalues = (s ** 2) / (num_components - 1)

    return eigenvalues[:num_components], V_T[:, :num_components]


def apply_batch_svd(centered_points, num_components):
    """
    Applies SVD (singular value decomposition) to a centered data matrix for each element in a batch and returns the
    first 'num_components' eigenvector/eigenvalue pairs for each batch element to create a point distribution model
    (PDM).

    :param centered_points: Centered data matrix with shape (num_points, num_samples, batch_size).
    :type centered_points: torch.Tensor
    :param num_components: Indicates how many eigenvector/eigenvalue pairs are considered (rank of the model).
    :type num_components: int
    :return: Tuple with two elements:
        - torch.Tensor: The first 'num_components' eigenvalues in descending order with shape
          (num_components, batch_size).
        - torch.Tensor: The corresponding first 'num_components' eigenvectors with shape
          (num_points, num_components, batch_size).
    """
    # Transpose to shape (batch_size, num_samples, num_points) for batched SVD
    centered_points = centered_points.permute(2, 1, 0)
    U, s, V_T = torch.linalg.svd(centered_points, full_matrices=False)
    eigenvalues = (s ** 2) / (num_components - 1)

    eigenvalues = eigenvalues[:, :num_components]
    V_T = V_T[:, :num_components, :]

    return eigenvalues.t(), V_T.permute(2, 1, 0)


def get_parameters(stacked_points, components):
    """
    Determines the model parameters in the model space that correspond to the given mesh points. This involves solving
    the least squares problem.

    :param stacked_points: Batch with given centered mesh points with shape (3 * num_points, batch_size).
    :type stacked_points: torch.Tensor
    :param components: Eigenvectors of the model multiplied by the square root of the respective eigenvalue with shape
    (3 * num_points, model_rank).
    :type components: torch.Tensor
    :return: Calculated model parameters with shape (model_rank, batch_size).
    :rtype: torch.Tensor
    """
    return torch.linalg.lstsq(components, stacked_points).solution


def get_batch_parameters(stacked_points, components):
    """
    Determines the model parameters in a batch of model spaces that correspond to the given mesh points. This involves
    solving least squares problems.

    :param stacked_points: Batch with given centered mesh points with shape (3 * num_points, batch_size, num_models).
    :type stacked_points: torch.Tensor
    :param components: Eigenvectors of the models multiplied by the square root of the respective eigenvalue with shape
    (3 * num_points, model_rank, num_models).
    :type components: torch.Tensor
    :return: Calculated model parameters with shape (model_rank, batch_size, num_models).
    :rtype: torch.Tensor
    """
    parameters = torch.linalg.lstsq(components.permute(2, 0, 1), stacked_points.permute(2, 0, 1)).solution
    return parameters.permute(1, 2, 0)


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


def unnormalised_log_gaussian_pdf(x, mean=0.0, sigma=1.0):
    """
    Calculates the unnormalised log-likelihood of points in 1-dimensional space assuming a Gaussian distribution,
    defined by the input parameters ‘mean’ and ‘covariance’. Supports batched input points.

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
    return torch.sum(-0.5 * ((x - mean) ** 2) / (sigma ** 2), dim=0)


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
    # TODO: Write docstring.
    """
    NOTE: Method is currently not used.

    :param ref_points:
    :param target_points:
    :param batch_size:
    :return:
    """
    target_points_expanded = target_points.unsqueeze(2).expand(-1, -1, batch_size).unsqueeze(0)
    points_expanded = ref_points.unsqueeze(1)
    distances = torch.sub(points_expanded, target_points_expanded)
    return torch.sqrt(torch.min(torch.sum(torch.pow(distances, float(2)), dim=2), dim=1)[0])
    # closest_points = torch.argmin(torch.sum(torch.pow(distances, float(2)), dim=2), dim=0)
    # Not sure what the following lines of code were supposed to do.
    # distances = distances[closest_points, torch.arange(1000).unsqueeze(1).expand(1000, 100), :,
    #            torch.arange(100)]
    # return torch.transpose(distances, 1, 2)


def index_of_closest_point(ref_points, target_points, batch_size):
    # TODO: Write docstring.
    """
    NOTE: Method is currently not used.

    :param ref_points:
    :param target_points:
    :param batch_size:
    :return:
    """
    target_points_expanded = target_points.unsqueeze(2).expand(-1, -1, batch_size).unsqueeze(0)
    points_expanded = ref_points.unsqueeze(1)
    distances = torch.sub(points_expanded, target_points_expanded)
    closest_idx = torch.argmin(torch.sum(torch.pow(distances, float(2)), dim=2), dim=1)
    return closest_idx


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
    likelihoods = batch_multivariate_gaussian_pdf(3, differences, torch.zeros(3, dtype=torch.float32),
                                                  torch.diag(sigma_lm * torch.ones(3, dtype=torch.float32)))
    log_likelihoods = torch.log(likelihoods)
    prior = gaussian_pdf(parameters, sigma=sigma_prior)
    log_prior = torch.log(prior)
    return torch.sum(torch.cat((log_likelihoods, log_prior), dim=0), dim=0)


class PointDistributionModel:
    def __init__(self, read_in=False, mean_and_cov=False, meshes=None, mean=None, cov=None, model=None, rank=None,
                 dev=None, tol=10e-6):
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
        :type mean: torch.Tensor
        :param cov: Full covariance matrix with shape (3 * num_points, 3 * num_points).
        :type cov: torch.Tensor
        :param model: Read-in PDM in the form of a Python dictionary. To get this dictionary, use
        'src/custom_io/H5ModelIO.py'.
        :type model: dict
        :param rank: Specifies how many main components are to be taken into account. Must only be specified
        explicitly in the case of ‘mean_and_cov=True’.
        :type rank: int
        :param dev: An object representing the device on which the tensor operations are or will be allocated. Must only
        be specified explicitly in the case of ‘read_in=True’. Otherwise, the device is derived from the input data.
        :type dev: torch.device
        """
        self.read_in = read_in
        self.mean_and_cov = mean_and_cov
        if self.read_in:
            self.meshes = None
            self.stacked_points = None
            # self.mean = (model.get('points').reshape(-1, order='F') + model.get('mean'))[:, np.newaxis]
            # did not work. Why?
            self.dev = dev
            self.mean = torch.tensor((model.get('points').reshape(-1, order='F'))[:, np.newaxis], device=self.dev)
            self.points_centered = None
            self.num_points = int(self.mean.shape[0] / 3)
            self.rank = model.get('basis').shape[1]
            self.eigenvalues = torch.tensor(model.get('var'), device=self.dev)
            self.eigenvectors = torch.tensor(model.get('basis'), device=self.dev)
            self.components = self.eigenvectors * torch.tensor(model.get('std'), device=self.dev)
            self.parameters = None
            self.decimated = False
        elif self.mean_and_cov:
            self.meshes = None
            self.stacked_points = None
            self.dev = mean.device
            self.mean = mean
            self.points_centered = None
            self.num_points = int(self.mean.shape[0] / 3)
            self.rank = rank
            eigenvalues, self.eigenvectors = apply_svd(cov, self.rank)
            self.eigenvalues = torch.sqrt((self.rank - 1) * eigenvalues)
            self.components = self.eigenvectors * torch.sqrt(self.eigenvalues)
            self.parameters = None
            self.decimated = False
        else:
            self.meshes = meshes
            self.stacked_points = extract_points(self.meshes)
            self.dev = self.stacked_points.device
            self.mean = torch.mean(self.stacked_points, dim=1)[:, None]
            self.points_centered = self.stacked_points - self.mean
            # Avoid explicit representation of the covariance matrix
            # self.covariance = np.cov(self.stacked_points)
            # Dimensionality of 3 hardcoded here!
            self.num_points = int(self.stacked_points.size()[0] / 3)
            self.rank = self.stacked_points.size()[1]
            # Eigenvectors are the columns of the 2-dimensional ndarray 'self.eigenvectors'
            self.eigenvalues, self.eigenvectors = apply_svd(self.points_centered, self.rank)
            self.components = self.eigenvectors * torch.sqrt(self.eigenvalues)
            # self.parameters = get_parameters(self.points_centered, self.eigenvectors)
            # self.parameters = get_parameters(self.points_centered, self.components)
            self.decimated = False
        # Eliminate components with an eigenvalue that is too small to prevent numerical instabilities
        self.rank = torch.sum(self.eigenvalues > tol).item()
        self.eigenvectors = self.eigenvectors[:, self.eigenvalues > tol]
        self.components = self.components[:, self.eigenvalues > tol]
        self.eigenvalues = self.eigenvalues[self.eigenvalues > tol]
        if not self.read_in and not self.mean_and_cov:
            self.parameters = get_parameters(self.points_centered, self.components)

    def get_eigenvalues(self):
        """
        Returns all ('rank' many) eigenvalues of the PDM in descending order.

        :return: Torch tensor with the eigenvalues in descending order with shape (rank,).
        :rtype: torch.Tensor
        """
        return self.eigenvalues

    def get_eigenvalue_k(self, k):
        """
        Returns the kth-largest eigenvalue of the PDM.

        :param k: Specifies which eigenvalue is to be returned.
        :type k: int
        :return: The kth-largest eigenvalue.
        :rtype: torch.Tensor
        """
        return self.eigenvalues[k]

    def get_components(self):
        """
        Returns all components of the PDM. The kth component of the PDM is defined here as the square root of the kth
        eigenvalue times the kth eigenvector.

        :return: All ('rank' many) components of the PDM with shape (3 * num_points, rank).
        :rtype: torch.Tensor
        """
        return self.components

    def get_component_k(self, k):
        """
        Returns the kth component of the PDM. The kth component of the PDM is defined here as the square root of the kth
        eigenvalue times the kth eigenvector.

        :param k: Specifies which component is to be returned.
        :type k: int
        :return: The kth component of the PDM with shape (3 * num_points,).
        :rtype: torch.Tensor
        """
        return self.components[:, k]

    def get_eigenvectors(self):
        """
        Returns all eigenvectors of the PDM.

        :return: All ('rank' many) eigenvectors of the PDM with shape (3 * num_points, rank).
        :rtype: torch.Tensor
        """
        return self.eigenvectors

    def get_eigenvector_k(self, k):
        """
        Returns the kth eigenvector of the PDM.

        :param k: Specifies which eigenvector is to be returned.
        :type k: int
        :return: The kth eigenvector of the PDM with shape (3 * num_points,).
        :rtype: torch.Tensor
        """
        return self.eigenvectors[:, k]

    def get_points_from_parameters(self, parameters):
        """
        Takes model parameters and calculates the coordinates of all points of the instance defined by these parameters.
        Works for a single set of model parameters as well as for entire batches of model parameters.

        :param parameters: Model parameters for which the point coordinates of the associated instance are to be
        calculated, either of shape (rank,) or (rank, batch_size).
        :type parameters: torch.Tensor
        :return: Calculated point coordinates, either of shape (num_points, 3) or (num_points, 3, batch_size).
        :rtype: torch.Tensor
        """
        if parameters.ndim == 1:
            parameters = parameters.unsqueeze(1)  # Equivalent to parameters[:, np.newaxis] in NumPy

        batch_size = parameters.size()[1]

        stacked_points = self.components @ parameters + self.mean

        if batch_size == 1:
            return stacked_points.reshape((-1, 3))
        else:
            return stacked_points.reshape((-1, 3, batch_size))

    def decimate(self, decimation_target=200):
        # TODO: Implement model decimation for all PDMs.
        """
        Reduces the PDM to the specified number of points. Currently, only calculated PDMs can be reduced, i.e. if
        ‘mean_and_cov'='read_in'=False.

        :param decimation_target: Number of target points that the reduced PDM should have.
        :type decimation_target: int
        :return: The first mesh is reduced to the number of target points using QEM, then the PDM is defined at these
        points using interpolation (radial basis functions). This mesh instance is returned. Nothing is returned if the
        decimation was not successful.
        :rtype: TorchMeshGpu
        """
        if self.read_in or self.mean_and_cov:
            warnings.warn("Warning: Decimation of imported Point Distribution Models is not (yet) supported.",
                          UserWarning)
            return None

        if self.num_points <= decimation_target:
            warnings.warn("Warning: No decimation necessary, as the target is below the current number of points.",
                          UserWarning)
            return None

        if self.decimated:
            warnings.warn("Warning: Point Distribution Model can only be decimated once.",
                          UserWarning)
            return None

        reference_decimated = self.meshes[0].simplify_qem(decimation_target)
        # reference_decimated = ref.simplify_qem(decimation_target)
        mean_fun, cov_fun = self.pdm_to_interpolated_gp(decimation_target)
        mean, cov = mean_fun(reference_decimated.tensor_points), cov_fun(reference_decimated.tensor_points,
                                                                         reference_decimated.tensor_points)
        self.mean = mean.reshape((-1, 1))
        eigenvalues, eigenvectors = apply_svd(cov, self.rank)
        self.eigenvalues = torch.sqrt(eigenvalues[:self.rank] * (self.rank - 1))
        self.eigenvectors = eigenvectors[:, :self.rank]
        self.meshes = None
        self.stacked_points = None
        self.points_centered = None
        self.num_points = reference_decimated.num_points
        self.components = self.eigenvectors * torch.sqrt(self.eigenvalues)
        self.parameters = None
        self.decimated = True

        return reference_decimated

    def pdm_to_interpolated_gp(self, decimation_target):
        """
        Extends the PDM, which is only defined at a finite number of points, to an infinite domain. This then
        corresponds to a Gaussian process (GP). The values of the deformation field between the defined points are
        interpolated using a radial basis functions (RBF) interpolator.
        Remark: The SciPy RBF interpolators converts Torch tensors internally to Numpy arrays. This is
        disadvantageous for the runtime. Each call of the mean or covariance function generates therefore a lot of
        traffic between the CPU and GPU when the rest of the code is running on the GPU. CuPy, for example, can be tried
        out later to optimise the code (needs Python 3.9 or later versions).

        :param decimation_target: Number of target points that the reduced PDM should have.
        :type decimation_target: int
        :return: A tuple containing two functions.
            - The first function is the mean function of the GP.
            - The second function is the covariance function of the GP.
        :rtype: Tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
        """
        mean_reshaped = self.mean.reshape((-1, 3))
        eigenvectors_reshaped = self.eigenvectors.reshape((self.num_points, 3, self.rank))
        # SciPy accepts Torch tensors, but converts them internally into NumPy arrays. The tensors must therefore be
        # located on the CPU.
        eigenvector_interpolators = [RBFInterpolator(mean_reshaped.cpu(), eigenvectors_reshaped[:, :, i].cpu()) for i in
                                     range(self.rank)]

        def mean(x):
            """
            Evaluates the mean function at a given set of points x.

            :param x: Points at which the covariance function is to be analysed with shape (num_points, 3).
            :type x: torch.Tensor
            :return: Mean function evaluated at all the points in x with shape (num_points, 3).
            :rtype: torch.Tensor
            """
            return torch.tensor(RBFInterpolator(mean_reshaped.cpu(), mean_reshaped.cpu())(x.cpu()),
                                dtype=torch.float32, device=self.mean.device)

        def cov(x, y, num_points=decimation_target):
            """
            Evaluates the covariance function at given set of points x, y. The number of points in x and y must be
            identical.

            :param x: Points at which the covariance function is to be analysed with shape (num_points, 3).
            :type x: torch.Tensor
            :param y: Points at which the covariance function is to be analysed with shape (num_points, 3).
            :type y: torch.Tensor
            :param num_points: Indicates how many points are present in the sets x, y.
            :type num_points: int
            :return: Full covariance matrix that determines the covariance for each point in x with each point from y
            [shape (3 * num_points, 3 * num_points)].
            :rtype: torch.Tensor
            """
            x_eigen = torch.stack(
                [torch.tensor(interp(x.cpu()), device=x.device) for interp in eigenvector_interpolators],
                dim=-1).reshape((3 * num_points, self.rank))

            y_eigen = torch.stack(
                [torch.tensor(interp(y.cpu()), device=y.device) for interp in eigenvector_interpolators],
                dim=-1).reshape((3 * num_points, self.rank))

            return ((x_eigen * self.eigenvalues) @ y_eigen.t()).float()

        return mean, cov

    def get_covariance(self):
        """
        Calculates and returns the full covariance matrix of the PDM. For reasons of storage efficiency, the full
        covariance matrix is not permanently stored.

        :return: Full covariance matrix with shape (3 * num_points, 3 * num_points).
        :rtype: torch.Tensor
        """
        if not self.read_in and not self.mean_and_cov and not self.decimated:
            return torch.cov(self.stacked_points)
        else:
            return self.eigenvectors @ (torch.diag(self.eigenvalues) @ self.eigenvectors.t())

    def change_device(self, dev):
        """
        Change the device on which the tensor operations are or will be allocated.

        :param dev: The future device on which the mesh data is to be saved.
        :type dev: torch.device
        """
        if self.dev == dev:
            return
        else:
            if not self.read_in and not self.mean_and_cov and not self.decimated:
                for mesh in self.meshes:
                    mesh.change_device(dev)
                self.stacked_points = self.stacked_points.to(dev)
                self.points_centered = self.points_centered.to(dev)
                self.parameters = self.parameters.to(dev)
            self.mean = self.mean.to(dev)
            self.eigenvalues = self.eigenvalues.to(dev)
            self.eigenvectors = self.eigenvectors.to(dev)
            self.components = self.components.to(dev)

            self.dev = dev

    def get_batched_pdm(self, batch_size):
        """
        Receive a batched version of this Point Distribution Model (PDM).

        :param batch_size: Specifies how many copies of the PDM should be present in the BatchedPointDistributionModel
        instance.
        :type batch_size: int
        :return: Said BatchedPointDistributionModel instance.
        :rtype: BatchedPointDistributionModel
        """
        return BatchedPointDistributionModel(batch_size=batch_size, mean_and_cov=True,
                                             mean=self.mean.expand(-1, batch_size),
                                             cov=self.get_covariance().unsqueeze(-1).expand(-1, -1, batch_size),
                                             rank=self.rank)


class BatchedPointDistributionModel(PointDistributionModel):
    def __init__(self, batch_size, read_in=False, mean_and_cov=False, meshes=None, mean=None, cov=None, model=None,
                 rank=None, dev=None):
        """
        Class that defines and creates a batched point distribution model (PDM), i.e., 'batch_size' many different PDMs
        with the same number of points and the same rank.
        This can be done in three different ways: It can be imported from a .h5 file, a complete set of mean vector and
        covariance matrix can be passed, or a batch of meshes can be given from which the PDM is to be calculated.
        If the model was read in from a .h5 file or was created from a list of meshes, the same model is copied
        ‘batch_size’ times.
        Remark: Of the two Boolean values ‘read_in’ and ‘mean_and_cov’, at most one should be True.

        :param batch_size: Specifies how many point distribution models are to be represented simultaneously.
        :type batch_size: int
        :param read_in: Boolean value that specifies whether a read in model is available.
        :type read_in: bool
        :param mean_and_cov: Boolean value that specifies whether a complete set of mean vector and covariance matrix is
        passed.
        :type mean_and_cov: bool
        :param meshes: List of meshes from which the PDM is to be calculated.
        :type meshes: list
        :param mean: Mean vector with shape (3 * num_points, batch_size).
        :type mean: torch.Tensor
        :param cov: Full covariance matrix with shape (3 * num_points, 3 * num_points, batch_size).
        :type cov: torch.Tensor
        :param model: Read-in PDM in the form of a Python dictionary. To get this dictionary, use
        'src/custom_io/H5ModelIO.py'.
        :type model: dict
        :param rank: Specifies how many main components are to be taken into account. Must only be specified
        explicitly in the case of ‘mean_and_cov=True’.
        :type rank: int
        :param dev: An object representing the device on which the tensor operations are or will be allocated. Must only
        be specified explicitly in the case of ‘read_in=True’. Otherwise, the device is derived from the input data.
        :type dev: torch.device
        """
        self.batch_size = batch_size
        self.read_in = read_in
        self.mean_and_cov = mean_and_cov
        if self.read_in:
            self.meshes = None
            self.stacked_points = None
            # self.mean = (model.get('points').reshape(-1, order='F') + model.get('mean'))[:, np.newaxis]
            # did not work. Why?
            self.dev = dev
            self.mean = torch.tensor((model.get('points').reshape(-1, order='F'))[:, np.newaxis], device=self.dev) \
                .expand(-1, self.batch_size)
            self.points_centered = None
            self.num_points = int(self.mean.shape[0] / 3)
            self.mean = self.mean.expand(-1, self.batch_size)
            self.rank = model.get('basis').shape[1]
            self.eigenvalues = torch.tensor(model.get('var'), device=self.dev).unsqueeze(-1).expand(-1, self.batch_size)
            self.eigenvectors = torch.tensor(model.get('basis'), device=self.dev).unsqueeze(-1).expand(-1, -1,
                                                                                                       self.batch_size)
            self.components = self.eigenvectors * torch.tensor(model.get('std'), device=self.dev).view(1, self.rank, 1)
            self.parameters = None
            self.decimated = False
        elif self.mean_and_cov:
            self.meshes = None
            self.stacked_points = None
            self.dev = mean.device
            self.mean = mean
            self.points_centered = None
            self.num_points = int(self.mean.shape[0] / 3)
            self.rank = rank
            eigenvalues, self.eigenvectors = apply_batch_svd(cov, self.rank)
            self.eigenvalues = torch.sqrt((self.rank - 1) * eigenvalues)
            self.components = self.eigenvectors * torch.sqrt(self.eigenvalues).view(1, self.rank, -1)
            self.parameters = None
            self.decimated = False
        else:
            self.meshes = meshes
            self.stacked_points = extract_points(self.meshes)
            self.dev = self.stacked_points.device
            self.mean = torch.mean(self.stacked_points, dim=1)[:, None]
            self.points_centered = (self.stacked_points - self.mean).unsqueeze(-1).expand(-1, -1, self.batch_size)
            self.mean = self.mean.expand(-1, self.batch_size)
            # Avoid explicit representation of the covariance matrix
            # self.covariance = np.cov(self.stacked_points)
            # Dimensionality of 3 hardcoded here!
            self.num_points = int(self.stacked_points.size()[0] / 3)
            self.rank = self.stacked_points.size()[1]
            # Eigenvectors are the columns of the 2-dimensional ndarray 'self.eigenvectors'
            self.eigenvalues, self.eigenvectors = apply_batch_svd(self.points_centered, self.rank)
            self.components = self.eigenvectors * torch.sqrt(self.eigenvalues).view(1, self.rank, -1)
            # self.parameters = get_parameters(self.points_centered, self.eigenvectors)
            self.parameters = get_batch_parameters(self.points_centered[:, :, :], self.components)
            self.decimated = False

    def get_component_k(self, k):
        """
        Returns the kth component of all the PDMs. The kth component of the PDM is defined here as the square root of
        the kth eigenvalue times the kth eigenvector.

        :param k: Specifies which component is to be returned.
        :type k: int
        :return: The kth component of the PDM with shape (3 * num_points, batch_size).
        :rtype: torch.Tensor
        """
        return self.components[:, k, :]

    def get_eigenvalue_k(self, k):
        """
        Returns the kth eigenvalue of all the PDMs.

        :param k: Specifies which eigenvector is to be returned.
        :type k: int
        :return: The kth eigenvector of the PDM with shape (3 * num_points, batch_size).
        :rtype: torch.Tensor
        """
        return self.components[:, k, :]

    def get_points_from_parameters(self, parameters):
        """
        Takes model parameters and calculates the coordinates of all points of the instance defined by these
        parameters. There are two options: Either only one set of model parameters is provided, or as many sets as the
        batch size of the batched PDM. In the first case, the same parameters are used as input for all internal PDMs.
        In the second case, the i-th set of model parameters is evaluated using the i-th model.

        :param parameters: Model parameters for which the point coordinates of the associated instances are to be
        calculated, either of shape (rank,) or (rank, batch_size).
        :type parameters: torch.Tensor
        :return: Calculated point coordinates of shape (num_points, 3, batch_size).
        :rtype: torch.Tensor
        """
        if parameters.ndim == 1:
            parameters = parameters.unsqueeze(1).expand(-1, self.batch_size)

        stacked_points = torch.bmm(self.components.permute(2, 0, 1), parameters.unsqueeze(0)
                                   .expand(self.batch_size, -1, -1)).permute(1, 0, 2) + self.mean.unsqueeze(2)

        return torch.diagonal(stacked_points.reshape((-1, 3, self.batch_size, self.batch_size)), dim1=2, dim2=3)

    def decimate(self, decimation_target=200):
        """
        This method is not available for BatchPointDistributionModel instances.

        :param decimation_target:
        """
        warnings.warn("Warning: PDM method invoked from BatchPDM instance. No action taken.",
                      UserWarning)

    def pdm_to_interpolated_gp(self, decimation_target):
        """
        This method is not available for BatchPointDistributionModel instances.

        :param decimation_target:
        """
        warnings.warn("Warning: PDM method invoked from BatchPDM instance. No action taken.",
                      UserWarning)

    def get_covariance(self):
        """
        Calculates and returns the full covariance matrix of the PDM. For reasons of storage efficiency, the full
        covariance matrix is not permanently stored.

        :return: Full covariance matrix with shape (3 * num_points, 3 * num_points).
        :rtype: torch.Tensor
        """
        if not self.read_in and not self.mean_and_cov and not self.decimated:
            return (torch.matmul(self.points_centered.permute(2, 0, 1), self.points_centered.permute(2, 1, 0)) /
                    self.points_centered.size()[1]).permute(1, 2, 0)
        else:
            return (self.eigenvectors.permute(2, 0, 1) @ (torch.diag_embed(self.eigenvalues.permute(1, 0)) @
                                                          self.eigenvectors.permute(2, 1, 0))).permute(1, 2, 0)

    def get_pdm(self, k):
        """
        Returns the kth model of the batched PDM as a single PDM.

        :param k: Specifies which model is to be extracted.
        :type k: int
        :return: Said PDM.
        :rtype: PointDistributionModel
        """
        return PointDistributionModel(mean_and_cov=True,
                                      mean=self.mean[:, k],
                                      cov=self.get_covariance()[:, :, k],
                                      rank=self.rank)
