import json
from pathlib import Path

import meshio

from src.mesh import TorchMeshGpu


def read_and_simplify_registered_meshes(read_path, write_path, dev, target=200):
    rel_read_path = Path(read_path)
    rel_write_path = Path(write_path)
    read_path = Path.cwd().parent / rel_read_path
    write_path = Path.cwd().parent / rel_write_path
    meshes = []
    simplified_meshes = []
    simplified_ref_full, simplified_ref_dec = None, None
    for idx, file in enumerate(read_path.iterdir()):
        if file.suffix == '.stl' or file.suffix == '.ply':
            mesh_read_io = MeshReaderWriter(file)
            mesh_write_io = MeshReaderWriter(write_path / file.name)
            mesh = mesh_read_io.read_mesh(dev)
            meshes.append(mesh)
            if idx == 0:
                simplified_ref_full = mesh
                simplified_mesh = mesh.simplify_qem(target)
                simplified_ref_dec = simplified_mesh
            else:
                simplified_mesh = mesh.simplify_ref(simplified_ref_full, simplified_ref_dec)
            mesh_write_io.write_mesh(simplified_mesh)
            simplified_meshes.append(simplified_mesh)
    return meshes, simplified_meshes


def read_meshes(relative_path, dev):
    relative_path = Path(relative_path)
    mesh_path = Path.cwd().parent / relative_path
    meshes = []
    meshes_io = []
    for file in mesh_path.iterdir():
        if file.suffix == '.stl' or file.suffix == '.ply':
            mesh_io = MeshReaderWriter(file)
            mesh = mesh_io.read_mesh(dev)
            meshes.append(mesh)
            meshes_io.append(mesh_io)
    return meshes, meshes_io


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

    def read_mesh(self, dev):
        mesh = meshio.read(self.file)
        torch_mesh = TorchMeshGpu.from_mesh(mesh, self.name, dev)
        return torch_mesh

    def write_mesh(self, mesh):
        write_path = self.file.parent
        write_path.mkdir(parents=True, exist_ok=True)
        file_path = write_path / mesh.id
        mesh.change_device('cpu')
        mesh.points = mesh.tensor_points
        meshio.write(file_path.with_suffix('.stl'), mesh)
