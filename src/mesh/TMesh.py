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
        return cls(mesh, identifier)

    def simplify_qem(self, target=1000):
        new_coordinates, new_triangles = simplify_mesh(self.points.astype(np.float64),
                                                       self.cells[0].data.astype(np.uint32), num_nodes=target)
        new_mesh = meshio.Mesh(new_coordinates.astype(np.float32), [meshio.CellBlock('triangle',
                                                                                     new_triangles.astype(np.int64))])
        # return self.from_mesh(new_mesh, self.id + '_simplified_' + str(target))
        return self.from_mesh(new_mesh, self.id)

    def new_mesh_from_transformation(self, transformed_points):
        # transformed_points: numpy_array
        copied_mesh = self.copy()
        copied_mesh.tensor_points = torch.tensor(transformed_points)
        copied_mesh.points = transformed_points
        return copied_mesh

    def apply_transformation(self, transform):
        additional_col = np.ones((self.points.shape[0], 1))
        extended_points = np.hstack((self.points, additional_col))
        transformed_points = extended_points @ transform.T
        transformed_points = transformed_points[:, :3]
        return self.new_mesh_from_transformation(transformed_points)


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

    @classmethod
    def from_mesh(cls, mesh, identifier, batch_size=100):
        return cls(mesh, identifier, batch_size=batch_size)

    def simplify_qem(self, target=1000):
        warnings.warn("Warning: TorchMesh method invoked from BatchTorchMesh instance. No action taken.", UserWarning)
        return

    def new_mesh_from_transformation(self, transformed_points):
        # transformed_points: numpy_array
        copied_mesh = self.copy()
        copied_mesh.tensor_points = torch.tensor(transformed_points)
        copied_mesh.points = transformed_points
        return copied_mesh

    def apply_transformation(self, transform):
        # points_to_transform = self.points[:, :, 0]
        additional_cols = np.ones((self.points.shape[0], 1, self.points.shape[2]))
        extended_points = np.concatenate((self.points, additional_cols), axis=1)
        transformed_points = np.empty_like(extended_points)
        for z in range(extended_points.shape[2]):
            transformed_points[:, :, z] = extended_points[:, :, z] @ transform.T
        transformed_points = transformed_points[:, :3, :]
        return self.new_mesh_from_transformation(transformed_points)

    def set_points(self, new_points):
        self.tensor_points = torch.tensor(new_points)
        self.points = new_points
