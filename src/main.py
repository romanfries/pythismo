from pathlib import Path

import numpy as np

import custom_io
from custom_io.MeshIO import MeshReaderWriter
from model.PointDistribution import PointDistributionModel, PDMMetropolisSampler
from registration.Procrustes import ProcrustesAnalyser
from src.mesh.TMesh import BatchTorchMesh
from src.registration.IterativeClosestPoints import ICPAnalyser
from src.sampling.proposals.GaussRandWalk import GaussianRandomWalkProposal
from visualization.DashViewer import MeshVisualizer, BatchMeshVisualizer, MainVisualizer


def run(mesh_path,
        landmark_path=None,
        reference_path=None,
        simplify_meshes=False,
        read_landmarks=False,
        align_meshes=False):
    if simplify_meshes:
        write_path = Path(mesh_path).parent / 'meshes-simplified'
        original_meshes, meshes = custom_io.read_and_simplify_meshes(mesh_path, write_path)
    else:
        meshes = custom_io.read_meshes(mesh_path)

    if read_landmarks:
        reference, landmarks = custom_io.read_landmarks_with_reference(reference_path, landmark_path)

    if read_landmarks and align_meshes:
        landmark_aligner = ProcrustesAnalyser(landmarks, reference)
        transforms, _, identifiers = landmark_aligner.generalised_procrustes_alignment()

        dict_transforms = {identifier: transform for transform, identifier in zip(transforms, identifiers)}
        joined_meshes_transforms = []
        for mesh in meshes:
            if mesh.id in dict_transforms:
                joined_meshes_transforms.append((mesh, dict_transforms[mesh.id],
                                                 mesh.id))

        transformed_meshes = []
        transformed_path = Path.cwd().parent / 'datasets' / 'femur-data' / 'project-data' / 'meshes-simplified' \
                                                                                            '-aligned'
        transformed_path.mkdir(parents=True, exist_ok=True)
        for mesh, transform, _ in joined_meshes_transforms:
            mesh.apply_transformation(transform)
            transformed_meshes.append(mesh)

        meshes = transformed_meshes

        icp_aligner = ICPAnalyser(meshes)
        icp_aligner.icp()

        for mesh, _, _ in joined_meshes_transforms:
            file_name = str(mesh.id) + '.stl'
            mesh_io = MeshReaderWriter(transformed_path / file_name)
            mesh_io.write_mesh(mesh)

    elif align_meshes:
        # Problem: Correspondences are not given!
        mesh_aligner = ProcrustesAnalyser(meshes, data_format="meshio_mesh")
        _, new_points, identifiers = mesh_aligner.generalised_procrustes_alignment()

        joined_meshes_new_points = []
        dict_new_points = {identifier: new_points for new_points, identifier in zip(new_points, identifiers)}
        for mesh in meshes:
            if mesh.id in dict_new_points:
                joined_meshes_new_points.append((mesh, dict_new_points[mesh.id],
                                                 mesh.id))

        transformed_meshes = []
        transformed_path = Path.cwd().parent / 'datasets' / 'femur-data' / 'project-data' / 'meshes-simplified-aligned'
        transformed_path.mkdir(parents=True, exist_ok=True)
        for mesh, new_points, _ in joined_meshes_new_points:
            mesh.set_points(new_points)
            transformed_meshes.append(mesh)
            file_name = str(mesh.id) + 'stl'
            mesh_io = MeshReaderWriter(transformed_path / file_name)
            mesh_io.write_mesh(mesh)

        meshes = transformed_meshes

    batch_meshes = []
    for mesh in meshes:
        batch_mesh = BatchTorchMesh.from_mesh(mesh, mesh.id)
        batch_meshes.append(batch_mesh)

    model = PointDistributionModel(meshes)
    for index, batch_mesh in enumerate(batch_meshes):
        if index == 0:
            random_walk = GaussianRandomWalkProposal(batch_mesh.batch_size, model.parameters[:, index])
            converter = PDMMetropolisSampler(model, random_walk, batch_mesh, meshes[46], correspondences=True)
            for i in range(2):
                converter.propose()
                converter.determine_quality()
                converter.decide()
                print(converter.acceptance_ratio())

    # random_walk = GaussianRandomWalkProposal(mesh)
    # random_walk.apply()

    visualizer = MainVisualizer(batch_meshes[0], model, converter)
    # print(converter.acceptance_ratio())
    visualizer.run()

    print('Successful')


class Main:
    def __init__(self):
        pass


if __name__ == "__main__":
    main = Main()
    run("datasets/femur-data/project-data/registered",
        landmark_path="datasets/femur-data/project-data/landmarks",
        reference_path="datasets/femur-data/project-data/reference-landmarks",
        simplify_meshes=False,
        read_landmarks=False,
        align_meshes=False)
