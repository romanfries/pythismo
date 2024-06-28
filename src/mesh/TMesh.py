import meshio
from meshio import Mesh
import torch


class MeshIOService:
    def __init__(self, reader=meshio.read, writer=meshio.write):
        self.reader = reader
        self.writer = writer

    def read_mesh(self, file_path, identifier):
        mesh = self.reader(file_path)
        torch_mesh = TorchMesh.from_mesh(mesh, identifier)
        return torch_mesh

    def write_mesh(self, torch_mesh, file_path):
        mesh = Mesh(torch_mesh)
        self.writer(file_path, mesh)


class TorchMesh(Mesh):
    def __init__(self, mesh, identifier):
        # possibly prone to errors
        args = list(vars(mesh).values())
        super().__init__(*args)
        self.tensor_points = torch.tensor(self.points)
        self.id = identifier

    @classmethod
    def from_mesh(cls, mesh, identifier):
        return cls(mesh, identifier)

    def new_mesh_from_perturbation(self, perturbed_points):
        copied_mesh = self.copy()
        copied_mesh.tensor_points = perturbed_points
        copied_mesh.points = perturbed_points.numpy()
        return copied_mesh

