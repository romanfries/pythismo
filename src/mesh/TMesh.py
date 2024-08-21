import numpy as np
import meshio
import pytorch3d
from meshio import Mesh
import torch
from quad_mesh_simplify import simplify_mesh
import warnings


def get_transformation_matrix(angles):
    """
    Generates suitable transformation matrices for given Euler angles (ZYX convention).
    The code is inspired by J. Gregson, but extended so that it can handle batches of input angles:
    https://gist.github.com/jamesgregson/67eb5509af0d8b372f25146d5e3c5149

    :param angles: Numpy array with (optional: a batch of) Euler coordinates in radians (shape (3,) or (3, batch_size)).
    :type angles: np.ndarray
    :return: Numpy array with (a batch of) transformation matrix/matrices (shape (4, 4) or (4, 4, batch_size)).
    :rtype: np.ndarray
    """
    if angles.ndim == 1:
        angles = angles[:, np.newaxis]
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)
    transformation_matrices = np.zeros((4, 4, angles.shape[1]))

    transformation_matrices[0, 0, :] = cy * cz
    transformation_matrices[0, 1, :] = -cx * sz + cz * sx * sy
    transformation_matrices[0, 2, :] = cx * cz * sy + sx * sz
    transformation_matrices[1, 0, :] = cy * sz
    transformation_matrices[0, 3, :] = 0

    transformation_matrices[1, 1, :] = cx * cz + sx * sy * sz
    transformation_matrices[1, 2, :] = cx * sy * sz - cz * sx
    transformation_matrices[2, 0, :] = -sy
    transformation_matrices[1, 3, :] = 0

    transformation_matrices[2, 1, :] = cy * sx
    transformation_matrices[2, 2, :] = cx * cy
    transformation_matrices[2, 3, :] = 0
    transformation_matrices[3, 3, :] = 1

    if angles.shape[1] == 1:
        transformation_matrices = np.squeeze(transformation_matrices)

    return transformation_matrices


class TorchMeshIOService:
    def __init__(self, reader=meshio.read, writer=meshio.write):
        self.reader = reader
        self.writer = writer

    def read_mesh(self, file_path, identifier):
        mesh = self.reader(file_path)
        torch_mesh = TorchMesh.from_mesh(mesh, identifier)
        return torch_mesh

    def write_mesh(self, torch_mesh, file_path):
        self.writer(file_path, torch_mesh)


class TorchMesh(Mesh):
    def __init__(self, mesh, identifier):
        """
        This class is an extension of the meshio.Mesh class. It extends the superclass with a torch.Tensor to perform
        mesh calculations using PyTorch. The dimensionality is fixed at 3.
        For more information on meshio, please visit https://github.com/nschloe/meshio.

        :param mesh: Mesh instance that serves as a base.
        :type mesh: meshio.Mesh
        :param identifier: Identification string.
        :type identifier: str

        """
        # possibly prone to errors
        arg_dict = vars(mesh)
        arg_dict = {key: value for key, value in arg_dict.items() if key not in {'tensor_points', 'id', 'num_points',
                                                                                 'dimensionality'}}
        args = list(arg_dict.values())
        super().__init__(*args)
        self.tensor_points = torch.tensor(self.points)
        self.num_points = self.points.shape[0]
        self.dimensionality = self.points.shape[1]
        self.id = identifier

    @classmethod
    def from_mesh(cls, mesh, identifier):
        """
        Redundant method.
        :param mesh:
        :param identifier:
        :return:
        """
        return cls(mesh, identifier)

    def simplify_qem(self, target=1000):
        """
        Reduces the mesh to the specified number of points and returns a new instance with these new points/triangles.
        The identifier remains unchanged.
        The algorithm used comes from the paper by M. Garland and P. Heckbert: "Surface Simplification Using Quadric
        Error Metrics" (1997) implemented by J. Magnusson. For more information, please visit
        https://github.com/jannessm/quadric-mesh-simplification


        :param target: Number of nodes that the final mesh will have.
        :type target: int
        :return: Reduced TorchMesh instance with the specified number of points/nodes.
        :rtype: TorchMesh
        """
        new_coordinates, new_triangles = simplify_mesh(self.points.astype(np.float64),
                                                       self.cells[0].data.astype(np.uint32), num_nodes=target)
        new_mesh = meshio.Mesh(new_coordinates.astype(np.float32), [meshio.CellBlock('triangle',
                                                                                     new_triangles.astype(np.int64))])
        # return self.from_mesh(new_mesh, self.id + '_simplified_' + str(target))
        return self.from_mesh(new_mesh, self.id)

    def set_points(self, transformed_points):
        """
        The method for changing the points of the mesh. It ensures that the information of the mesh instance remains
        consistent, i.e. the same points are saved in both point fields (points, tensor_points).  The cells (triangles)
        remain unchanged.

        :param transformed_points: The new coordinates of the points. The shape of the
        np.ndarray has to be (num_points, 3)
        :type transformed_points: np.ndarray
        """
        # transformed_points: numpy_array
        self.tensor_points = torch.tensor(transformed_points)
        self.points = transformed_points

    def apply_transformation(self, transform, save_old=False):
        """
        Changes the points of the mesh by multiplying them by a (4, 4) transformation matrix.

        :param transform: Transformation matrix with shape (4, 4).
        :type transform: np.ndarray
        :param save_old: Boolean value that specifies whether the old point coordinates should be saved. Note: This is
        only relevant if the method is called from a BatchTorchMesh instance. TorchMesh does not offer the option of
        saving the old coordinates.
        :type save_old: bool
        """
        additional_col = np.ones((self.points.shape[0], 1))
        extended_points = np.hstack((self.points, additional_col))
        transformed_points = extended_points @ transform.T
        transformed_points = transformed_points[:, :3]
        self.set_points(transformed_points)

    def apply_translation(self, translation_parameters):
        """
        Changes the points of the mesh by adding a (3,) translation vector.

        :param translation_parameters: Array with shape (3,) containing the translation parameters.
        :type translation_parameters: np.ndarray
        """
        translated_points = self.points + translation_parameters
        self.set_points(translated_points)

    def apply_rotation(self, rotation_parameters, save_old=False):
        """
        Rotates the mesh instance using given Euler angles (ZYX convention).

        :param rotation_parameters: Array with shape (3,) containing the Euler angles (in radians).
        :type rotation_parameters: np.ndarray
        :param save_old: Boolean value that specifies whether the old point coordinates should be saved. Note: This is
        only relevant if the method is called from a BatchTorchMesh instance. TorchMesh does not offer the option of
        saving the old coordinates.
        :type save_old: bool
        """
        transformation = get_transformation_matrix(rotation_parameters)
        self.apply_transformation(transformation, save_old)

    def center_of_mass(self):
        """
        Calculates the centre of mass of the mesh object.

        :return: Coordinates of the centre of mass with shape (3,)
        :rtype: np.ndarray
        """
        return np.sum(self.points, axis=0) / self.num_points

    def to_pytorch3d_pointclouds(self, batch_size=1):
        """
        Helper method to translate a given TorchMesh into a pytorch3d.structures.Pointclouds object.
        The Pointclouds class provides functions for working with batches of 3d point clouds, and converting between
        representations.
        The generated Pointclouds object represents ‘batch_size’ times the same point cloud, namely the one consisting
        of the points of the calling TorchMesh instance.
        :param batch_size: Specifies how often the point cloud of the TorchMesh is repeated in the Pointclouds object.
        :type batch_size: int
        :return: Above described Pointclouds object.
        :rtype: pytorch3d.structures.Pointclouds
        """
        points = self.tensor_points.unsqueeze(0).repeat(batch_size, 1, 1)
        return pytorch3d.structures.Pointclouds(points)

    def calc_facet_normals(self):
        """
        (Re-)Calculates the normalised normal vectors to the triangular surfaces of the mesh.
        """
        triangles = torch.tensor(self.cells[0].data)
        v0, v1, v2 = self.tensor_points[triangles].unbind(dim=1)
        edges_a, edges_b = v1 - v0, v2 - v0
        facet_normals = torch.cross(edges_a, edges_b)
        facet_normals = facet_normals / torch.norm(facet_normals, dim=1, keepdim=True)
        self.cell_data.update({"facet_normals": facet_normals})


class BatchTorchMesh(TorchMesh):
    def __init__(self, mesh, identifier, batch_size=50, batched_data=False, batched_points=None):
        """
        Further extension of the TorchMesh class. Saves 'batch_size' many mesh instances simultaneously.
        However, these meshes differ only in the point coordinates and all have the same triangulation.
        There are two types of instantiation. Either only a TorchMesh is given, whose point coordinates are then
        replicated accordingly, or a batch of point coordinates.

        :param mesh: TorchMesh instance, which serves as the basis. The triangle data is transferred from it.
        :type mesh: TorchMesh
        :param identifier: Identification string.
        :type identifier: str
        :param batch_size: Self-explanatory.
        :type batch_size: int
        :param batched_data: Boolean variable that specifies whether the BatchTorchMesh is initialized with a batch of
        point coordinates.
        :type batched_data: bool
        :param batched_points: 3-dimensional array with shape (num_points, dimensionality, batch_size) with the batched
        point coordinates.
        :type batched_points: np.ndarray
        """
        # possibly prone to errors
        super().__init__(mesh, identifier)
        if not batched_data:
            self.tensor_points = self.tensor_points.unsqueeze(2).repeat(1, 1, batch_size)
            self.points = np.repeat(self.points[:, :, np.newaxis], batch_size, axis=2)
            self.batch_size = batch_size
        else:
            self.tensor_points = torch.tensor(batched_points)
            self.points = batched_points
            self.batch_size = self.points.shape[2]
        self.old_points = None

    @classmethod
    def from_mesh(cls, mesh, identifier, batch_size=100):
        """
        Redundant method.
        :param mesh:
        :param identifier:
        :return:
        """
        return cls(mesh, identifier, batch_size=batch_size)

    def simplify_qem(self, target=1000):
        """
        This method should not be used for BatchTorchMesh instances.

        :param target:
        """
        warnings.warn("Warning: TorchMesh method invoked from BatchTorchMesh instance. No action taken.", UserWarning)

    def set_points(self, transformed_points, save_old=False):
        """
        The method for changing the points of the mesh. It ensures that the information of the mesh instance remains
        consistent, i.e. the same points are saved in both point fields (points, tensor_points).  The cells (triangles)
        remain unchanged.

        :param transformed_points: The new coordinates of the points. The shape of the
        np.ndarray has to be (num_points, 3, batch_size)
        :type transformed_points: np.ndarray
        :param save_old: Boolean value that specifies whether the old point coordinates should be saved.
        :type save_old: bool
        """
        # transformed_points: numpy_array
        if save_old:
            self.old_points = self.tensor_points
        else:
            self.old_points = None
        self.tensor_points = torch.tensor(transformed_points)
        self.points = transformed_points

    def apply_transformation(self, transform, save_old=False):
        """
        Changes the points of the mesh by multiplying them by a (4, 4) transformation matrix.
        A separate transformation matrix must be provided for each element of the batch.
        Bear this in mind if the same transformation is to be applied to all elements of the batch by extending the
        'transform' array accordingly, e.g., by using np.newaxis.

        :param transform: Transformation matrix with shape (4, 4, batch_size).
        :type transform: np.ndarray
        :param save_old: Boolean value that specifies whether the old point coordinates should be saved.
        :type save_old: bool
        """
        # points_to_transform = self.points[:, :, 0]
        additional_cols = np.ones((self.points.shape[0], 1, self.points.shape[2]))
        extended_points = np.concatenate((self.points, additional_cols), axis=1)
        transformed_points = np.einsum('ijk,ljk->lik', transform, extended_points)
        transformed_points = transformed_points[:, :3, :]
        if save_old:
            self.old_points = self.tensor_points
        else:
            self.old_points = None
        self.set_points(transformed_points, save_old)

    def apply_translation(self, translation_parameters, save_old=False):
        """
        Changes the points of the mesh by adding a (3,) translation vector.
        A separate translation vector must be provided for each element of the batch.
        Bear this in mind if the same translation is to be applied to all elements of the batch by extending the
        'translation_parameters' array accordingly, e.g., by using np.newaxis.

        :param translation_parameters: Array with shape (3, batch_size) containing the translation parameters.
        :type translation_parameters: np.ndarray
        :param save_old: Boolean value that specifies whether the old point coordinates should be saved.
        :type save_old: bool
        """
        translated_points = self.points + translation_parameters
        if save_old:
            self.old_points = self.tensor_points
        else:
            self.old_points = None
        self.set_points(translated_points, save_old)

    def update_points(self, decider):
        """
        Internal method that updates the point values according to the information from the passed decider.

        :param decider: Boolean tensor of the shape (batch_size,), which indicates for each element of the batch whether
        the new point values were accepted (=True) or rejected (=False).
        :type decider: torch.Tensor
        """
        self.tensor_points = torch.where(decider.unsqueeze(0).unsqueeze(1), self.tensor_points,
                                         self.old_points)
        self.points = self.tensor_points.numpy()
        self.old_points = None

    def to_pytorch3d_meshes(self):
        """
        Helper method to translate a given BatchTorchMesh into a pytorch3d.structures.Meshes object.
        The Meshes class provides functions for working with batches of triangulated meshes with varying numbers of
        faces and vertices, and converting between representations.
        However, the generated Meshes object here is more restricted, because the meshes in the BatchTorchMesh instance
        all have the same number of points and the exact same triangulation.

        :return: Above described Meshes object.
        :rtype: pytorch3d.structures.Meshes
        """
        verts = torch.permute(self.tensor_points, (2, 0, 1))
        tensor_cells = torch.tensor(self.cells_dict['triangle'])
        faces = tensor_cells.unsqueeze(0).repeat(self.batch_size, 1, 1)
        return pytorch3d.structures.Meshes(verts, faces)

    def calc_facet_normals(self):
        """
        This method should not be used for BatchTorchMesh instances. The normal vectors are different for all elements
        of the batch.
        """
        warnings.warn("Warning: TorchMesh method invoked from BatchTorchMesh instance. No action taken.", UserWarning)
