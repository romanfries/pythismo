from pathlib import Path

from src.mesh.TMesh import TorchMeshIOService


class MeshReaderWriter:
    def __init__(self, file):
        self.file = Path(file).resolve()
        self.name = self.file.stem
        self.service = TorchMeshIOService()

    def read_mesh(self):
        mesh = self.service.read_mesh(self.file, self.name)
        return mesh

    def write_mesh(self, mesh):
        write_path = self.file.parents[1] / 'meshes-simplified'
        write_path.mkdir(parents=True, exist_ok=True)
        file_path = write_path / mesh.id
        self.service.write_mesh(mesh, file_path.with_suffix('.stl'))
