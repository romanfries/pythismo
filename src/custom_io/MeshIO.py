import re
from src.mesh.TMesh import MeshIOService, TorchMesh


class MeshReader:
    def __init__(self, file, identifier=None):
        self.file = file
        if identifier is None:
            self.identifier = re.sub(r'\.[^.]*$', '', file)
        else:
            self.identifier = identifier

    def read_mesh(self):
        mesh_service = MeshIOService()
        mesh = mesh_service.read_mesh(self.file, self.identifier)
        return mesh
