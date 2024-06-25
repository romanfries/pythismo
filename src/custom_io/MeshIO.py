import meshio
from src.mesh.TMesh import MeshIOService, TorchMesh


class MeshReader:
    def __init__(self, file):
        self.file = file

    def read_mesh(self):
        mesh_service = MeshIOService()
        mesh = mesh_service.read_mesh(self.file)
        return mesh
