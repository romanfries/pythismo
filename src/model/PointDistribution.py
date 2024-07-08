import numpy as np


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


def get_parameters(centered_points, eigenvectors):
    return np.transpose(eigenvectors) @ centered_points


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
    def __init__(self, pdm, proposal, batch_mesh):
        self.model = pdm
        self.proposal = proposal
        self.batch_mesh = batch_mesh

    def update_mesh(self):
        new_points = self.model.get_eigenvectors() @ self.proposal.parameters.numpy()
        new_points = new_points.reshape((self.batch_mesh.num_points, self.batch_mesh.dimensionality,
                                         self.proposal.batch_size))
        self.batch_mesh.set_points(new_points)
        return self.batch_mesh
