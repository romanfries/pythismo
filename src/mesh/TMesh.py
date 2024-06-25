import meshio
from meshio import Mesh
import torch


class MeshIOService:
    def __init__(self, reader=meshio.read, writer=meshio.write):
        self.reader = reader
        self.writer = writer

    def read_mesh(self, file_path):
        mesh = self.reader(file_path)
        torch_mesh = TorchMesh.from_mesh(mesh)
        return torch_mesh

    def write_mesh(self, torch_mesh, file_path):
        mesh = Mesh(torch_mesh)
        self.writer(file_path, mesh)


class TorchMesh(Mesh):
    def __init__(self, mesh):
        # possibly prone to errors
        args = list(vars(mesh).values())
        super().__init__(*args)
        self.tensor_points = torch.tensor(self.points)

    @classmethod
    def from_mesh(cls, mesh):
        return cls(mesh)

