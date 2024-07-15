import json
from pathlib import Path

from src.mesh.TMesh import TorchMeshIOService


def read_and_simplify_meshes(read_path, write_path, target=200):
    rel_read_path = Path(read_path)
    rel_write_path = Path(write_path)
    read_path = Path.cwd().parent / rel_read_path
    write_path = Path.cwd().parent / rel_write_path
    meshes = []
    simplified_meshes = []
    for file in read_path.iterdir():
        if file.suffix == '.stl' or file.suffix == '.ply':
            mesh_read_io = MeshReaderWriter(file)
            mesh_write_io = MeshReaderWriter(write_path / file.name)
            mesh = mesh_read_io.read_mesh()
            meshes.append(mesh)
            simplified_mesh = mesh.simplify_qem(target)
            mesh_write_io.write_mesh(simplified_mesh)
            simplified_meshes.append(simplified_mesh)
    return meshes, simplified_meshes


def read_meshes(relative_path):
    relative_path = Path(relative_path)
    mesh_path = Path.cwd().parent / relative_path
    meshes = []
    for file in mesh_path.iterdir():
        if file.suffix == '.stl' or file.suffix == '.ply':
            mesh_io = MeshReaderWriter(file)
            mesh = mesh_io.read_mesh()
            meshes.append(mesh)
    return meshes


def read_landmarks_with_reference(reference_path, landmark_path):
    rel_reference_path = Path(reference_path)
    rel_landmark_path = Path(landmark_path)
    reference_path = Path.cwd().parent / rel_reference_path
    landmark_path = Path.cwd().parent / rel_landmark_path

    landmarks = []
    for file in landmark_path.iterdir():
        if file.suffix == '.json':
            f = open(file)
            landmark = json.load(f)
            landmarks.append((landmark, file.stem))
            f.close()

    for file in reference_path.iterdir():
        if file.suffix == '.json':
            # Opening JSON file
            f = open(file)
            reference = json.load(f)
            f.close()

    return reference, landmarks


class MeshReaderWriter:
    def __init__(self, file):
        self.file = Path(file).resolve()
        self.name = self.file.stem
        self.service = TorchMeshIOService()

    def read_mesh(self):
        mesh = self.service.read_mesh(self.file, self.name)
        return mesh

    def write_mesh(self, mesh):
        write_path = self.file.parent
        write_path.mkdir(parents=True, exist_ok=True)
        file_path = write_path / mesh.id
        self.service.write_mesh(mesh, file_path.with_suffix('.stl'))
