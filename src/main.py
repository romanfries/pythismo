import json
import os
from pathlib import Path, PurePath

from custom_io.MeshIO import MeshReaderWriter
from registration.Procrustes import ProcrustesAnalyser
from sampling.proposals.GaussRandWalk import GaussianRandomWalkProposal
from visualization.DashViewer import ProposalVisualizer


def run(simplify_meshes=True):
    if simplify_meshes:
        mesh_path = Path.cwd().parent / 'datasets' / 'femur-data' / 'project-data' / 'meshes'
        meshes = []
        simplified_meshes = []
        for file in mesh_path.iterdir():
            if file.suffix == '.stl' or file.suffix == '.ply':
                mesh_io = MeshReaderWriter(file)
                mesh = mesh_io.read_mesh()
                meshes.append(mesh)
                simplified_mesh = mesh.simplify_qem()
                mesh_io.write_mesh(simplified_mesh)
                simplified_meshes.append(simplified_mesh)

        landmark_path = Path.cwd().parent / 'datasets' / 'femur-data' / 'project-data' / 'landmarks'
        landmarks = []
        for file in landmark_path.iterdir():
            if file.suffix == '.json':
                # Opening JSON file
                f = open(file)
                landmark = json.load(f)
                landmarks.append((landmark, file.stem))
                f.close()

        landmark_ref_path = Path.cwd().parent / 'datasets' / 'femur-data' / 'project-data' / 'reference-landmarks'
        for file in landmark_ref_path.iterdir():
            if file.suffix == '.json':
                # Opening JSON file
                f = open(file)
                landmark_ref = json.load(f)
                f.close()

        landmark_aligner = ProcrustesAnalyser(landmark_ref, landmarks)
        transforms, _, identifiers = landmark_aligner.generalised_procrustes_alignment()

        dict_transforms = {identifier: transform for transform, identifier in zip(transforms, identifiers)}
        joined_meshes_transforms = []
        for simplified_mesh in simplified_meshes:
            if simplified_mesh.id in dict_transforms:
                joined_meshes_transforms.append((simplified_mesh, dict_transforms[simplified_mesh.id],
                                                 simplified_mesh.id))

        transformed_meshes = []
        transformed_path = Path.cwd().parent / 'datasets' / 'femur-data' / 'project-data' / 'meshes-simplified-aligned'
        transformed_path.mkdir(parents=True, exist_ok=True)
        for mesh, transform, _ in joined_meshes_transforms:
            transformed_mesh = mesh.apply_transformation(transform)
            transformed_meshes.append(transformed_mesh)
            file_name = str(transformed_mesh.id) + 'stl'
            mesh_io = MeshReaderWriter(transformed_path / file_name)
            mesh_io.write_mesh(transformed_mesh)

    else:
        # TODO
        pass

    # random_walk = GaussianRandomWalkProposal(mesh)
    # random_walk.apply()

    # visualizer = ProposalVisualizer(random_walk)
    # visualizer.run()

    print('Successful')


class Main:
    def __init__(self):
        pass


if __name__ == "__main__":
    main = Main()
    run()
