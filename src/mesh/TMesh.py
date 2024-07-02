import numpy as np
import meshio
from meshio import Mesh
import torch
from quad_mesh_simplify import simplify_mesh


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
    def __init__(self, mesh, identifier, simplified=False):
        # possibly prone to errors
        args = list(vars(mesh).values())
        super().__init__(*args)
        self.tensor_points = torch.tensor(self.points)
        self.id = identifier
        self.simplified = simplified

    @classmethod
    def from_mesh(cls, mesh, identifier):
        return cls(mesh, identifier)

    def simplify_qem(self, target=1000):
        new_coordinates, new_triangles = simplify_mesh(self.points.astype(np.float64),
                                                       self.cells[0].data.astype(np.uint32), num_nodes=target)
        new_mesh = meshio.Mesh(new_coordinates.astype(np.float32), [meshio.CellBlock('triangle',
                                                                                     new_triangles.astype(np.int64))])
        return self.from_mesh(new_mesh, self.id + '_simplified_' + str(target))

    def new_mesh_from_transformation(self, transformed_points):
        copied_mesh = self.copy()
        copied_mesh.tensor_points = transformed_points
        copied_mesh.points = transformed_points.numpy()
        return copied_mesh

    def apply_transformation(self, transform):
        additional_col = np.ones((self.points.shape[0], 1))
        extended_points = np.hstack((self.points, additional_col))
        transformed_points = extended_points @ transform
        return self.new_mesh_from_transformation(transformed_points)
