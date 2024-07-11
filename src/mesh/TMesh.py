import numpy as np
import meshio
from meshio import Mesh
import torch
from quad_mesh_simplify import simplify_mesh
import warnings


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
        Reduces the mesh to the specified number of points and returns a new instance with these new points. The
        identifier remains unchanged.
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
        :return: None
        :rtype: None
        """
        # transformed_points: numpy_array
        self.tensor_points = torch.tensor(transformed_points)
        self.points = transformed_points

    def apply_transformation(self, transform):
        """
        Changes the points of the mesh by multiplying them by a (4, 4) transformation matrix.

        :param transform: Transformation matrix with shape (4, 4).
        :type transform: np.ndarray
        :return: None
        :rtype: None
        """
        additional_col = np.ones((self.points.shape[0], 1))
        extended_points = np.hstack((self.points, additional_col))
        transformed_points = extended_points @ transform.T
        transformed_points = transformed_points[:, :3]
        self.set_points(transformed_points)

    def center_of_mass(self):
        """
        Calculates the centre of mass of the mesh object.

        :return: Coordinates of the centre of mass with shape (3,)
        :rtype: np.ndarray
        """
        return np.sum(self.points, axis=0) / self.num_points


class BatchTorchMesh(TorchMesh):
    def __init__(self, mesh, identifier, batch_size=100, batched_data=False, batched_points=None):
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
        return cls(mesh, identifier, batch_size=batch_size)

    def simplify_qem(self, target=1000):
        """
        This method should not be used for BatchTorchMesh instances.

        :param target:
        :return:
        """
        warnings.warn("Warning: TorchMesh method invoked from BatchTorchMesh instance. No action taken.", UserWarning)
        return

    def set_points(self, transformed_points, save_old=False):
        """
        The method for changing the points of the mesh. It ensures that the information of the mesh instance remains
        consistent, i.e. the same points are saved in both point fields (points, tensor_points).  The cells (triangles)
        remain unchanged.

        :param transformed_points: The new coordinates of the points. The shape of the
         np.ndarray has to be (num_points, 3, batch_size)
        :type transformed_points: np.ndarray
        :param save_old:
        :type save_old:
        :return: None
        :rtype: None
        """
        # transformed_points: numpy_array
        if save_old:
            self.old_points = self.tensor_points
        else:
            self.old_points = None
        self.tensor_points = torch.tensor(transformed_points)
        self.points = transformed_points

    def apply_transformation(self, transform):
        """
        Changes the points of the mesh by multiplying them by a (4, 4) transformation matrix.

        :param transform: Transformation matrix with shape (4, 4).
        :type transform: np.ndarray
        :return: None
        :rtype: None
        """
        # points_to_transform = self.points[:, :, 0]
        additional_cols = np.ones((self.points.shape[0], 1, self.points.shape[2]))
        extended_points = np.concatenate((self.points, additional_cols), axis=1)
        transformed_points = np.empty_like(extended_points)
        for z in range(extended_points.shape[2]):
            transformed_points[:, :, z] = extended_points[:, :, z] @ transform.T
        transformed_points = transformed_points[:, :3, :]
        self.set_points(transformed_points)

    def update_points(self, decider):
        self.tensor_points = torch.where(decider.unsqueeze(0).unsqueeze(1), self.tensor_points,
                                         self.old_points)
        self.points = self.tensor_points.numpy()
        self.old_points = None
