from pathlib import Path

import custom_io
from custom_io.MeshIO import MeshReaderWriter
from model.PointDistribution import PointDistributionModel
from registration.Procrustes import ProcrustesAnalyser
from visualization.DashViewer import MeshVisualizer


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
                                                                                            '-aligned-new'
        transformed_path.mkdir(parents=True, exist_ok=True)
        for mesh, transform, _ in joined_meshes_transforms:
            transformed_mesh = mesh.apply_transformation(transform)
            transformed_meshes.append(transformed_mesh)
            file_name = str(transformed_mesh.id) + '.stl'
            mesh_io = MeshReaderWriter(transformed_path / file_name)
            mesh_io.write_mesh(transformed_mesh)

        meshes = transformed_meshes

    elif align_meshes:
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
            transformed_mesh = mesh.new_mesh_from_transformation(new_points)
            transformed_meshes.append(transformed_mesh)
            file_name = str(transformed_mesh.id) + 'stl'
            mesh_io = MeshReaderWriter(transformed_path / file_name)
            mesh_io.write_mesh(transformed_mesh)

        meshes = transformed_meshes


    # random_walk = GaussianRandomWalkProposal(mesh)
    # random_walk.apply()

    visualizer = MeshVisualizer(meshes)
    visualizer.run()

    model = PointDistributionModel(meshes)





    print('Successful')


class Main:
    def __init__(self):
        pass


if __name__ == "__main__":
    main = Main()
    run("datasets/femur-data/project-data/meshes-simplified-aligned",
        landmark_path="datasets/femur-data/project-data/landmarks",
        reference_path="datasets/femur-data/project-data/reference-landmarks",
        simplify_meshes=False,
        read_landmarks=False,
        align_meshes=False)
