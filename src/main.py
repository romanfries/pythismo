from pathlib import Path

import os

import meshio
import numpy as np
from meshio import Mesh

import custom_io
from model.PointDistribution import PointDistributionModel
from src.sampling.Metropolis import PDMMetropolisSampler
from src.custom_io.H5ModelIO import ModelReader
from src.mesh.TMesh import BatchTorchMesh, TorchMesh
from src.sampling.proposals.ClosestPoint import ClosestPointProposal
from src.sampling.proposals.GaussRandWalk import GaussianRandomWalkProposal, ParameterProposalType
from visualization.DashViewer import MainVisualizer


# TODO: These methods should be moved to src.mesh.TMesh.
def create_artificial_partial_target(full_target):
    """
    Creates a partial target, whereby all points above a defined plane are removed from a given mesh instance.
    The points defining the plane were determined by hand and are based on empirical values for the specific femur case.

    :param full_target: Original complete mesh instance.
    :type full_target: TorchMesh
    :return: Artificial partial target.
    :rtype: TorchMesh
    """
    p1, p2, p3 = np.array([0.0, 1.0, 100.0]), np.array([1.0, 0.0, 100.0]), np.array([0.0, 0.0, 100.0])
    normal, d = define_plane_from_points(p1, p2, p3)
    filtered_points, removed_indices = remove_points_above_plane(full_target.points, normal, d)
    filtered_cells = remove_triangles_with_removed_points(full_target.cells[0].data, removed_indices)
    filtered_cells = update_triangle_indices(full_target.cells[0].data, filtered_cells, removed_indices)
    cell_block = meshio.CellBlock('triangle', filtered_cells)
    return TorchMesh(Mesh(filtered_points, [cell_block]), 'target')


def define_plane_from_points(p1, p2, p3):
    """
    Defines a plane consisting of three points.

    :param p1: First point on the plane (np.ndarray of shape (3,)).
    :type p1: np.ndarray
    :param p2: Second point on the plane (np.ndarray of shape (3,)).
    :type p2: np.ndarray
    :param p3: Third point on the plane (np.ndarray of shape (3,)).
    :type p3: np.ndarray
    :return: Tuple with two elements:
        - np.ndarray: Normal vector of the plane with shape (3,).
        - float: Constant d in the plane equation.
    """
    v1 = p2 - p1
    v2 = p3 - p1

    normal = np.cross(v1, v2)
    d = -np.dot(normal, p1)

    return normal, d


def remove_points_above_plane(points, normal, d):
    """
    Removes all points from the array that lie above the defined level.

    :param points: Numpy array of points with shape (num_points, 3).
    :type points: np.ndarray
    :param normal: Normal vector of the plane with shape (3,).
    :type normal: np.ndarray
    :param d: Constant d in the plane equation.
    :type d: float
    :return: Tuple with two elements:
        - np.ndarray: Filtered array of points that are not above the plane with shape (num_filtered_points, 3)
        - np.ndarray: Indices of the removed points (in the 'points' array) with shape (num_removed_points,).
    """
    distances = np.dot(points, normal) + d
    mask = distances <= 0

    filtered_points = points[mask]
    removed_indices = np.where(~mask)[0]

    return filtered_points, removed_indices


def remove_triangles_with_removed_points(triangles, removed_indices):
    """
    Removes all triangles that contain at least one removed point.

    :param triangles: Numpy array of triangles with shape (num_triangles, 3).
    :type triangles: np.ndarray
    :param removed_indices: Indices of the removed points with shape (num_removed_points,).
    :type removed_indices: np.ndarray
    :return: Filtered array of triangles that do not contain any removed points with shape (num_filtered_triangles, 3).
    :rtype: np.ndarray
    """
    removed_indices_set = set(removed_indices)
    mask = np.array([not any(vertex in removed_indices_set for vertex in triangle) for triangle in triangles])

    filtered_triangles = triangles[mask]

    return filtered_triangles


def update_triangle_indices(triangles, filtered_triangles, removed_indices):
    """
    Updates the triangle data after points have been removed.

    :param triangles: Numpy array of original triangles (num_triangles, 3).
    :type triangles: np.ndarray
    :param filtered_triangles: Filtered numpy array of triangles with shape (num_filtered_triangles, 3).
    :type filtered_triangles: np.ndarray
    :param removed_indices: Indices of the removed points with shape (num_removed_points,).
    :type removed_indices: np.ndarray
    :return: Updated triangle data with shape (num_filtered_triangles, 3).
    :rtype: np.ndarray
    """
    index_mapping = np.zeros(np.max(triangles) + 1, dtype=int)
    index_mapping[removed_indices] = -1
    current_index = 0

    for i in range(len(index_mapping)):
        if index_mapping[i] != -1:
            index_mapping[i] = current_index
            current_index += 1

    updated_triangles = np.array([[index_mapping[vertex] for vertex in triangle] for triangle in filtered_triangles])

    return updated_triangles


def run(mesh_path,
        reference_path=None,
        model_path=None,
        read_model=False,
        simplify_model=False):
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    if read_model:
        rel_model_path = Path(model_path)
        model_path = Path.cwd().parent / rel_model_path
        model_reader = ModelReader(model_path)
        reference = custom_io.read_meshes(reference_path)[0]
        model = model_reader.get_model()
        # TODO: Think again: Does it make sense to represent the target as a TorchMesh?
        target = reference.copy()
        target.set_points(model.get_points_from_parameters(1.0 * np.ones(model.sample_size)))
        reference.set_points(model.get_points_from_parameters(np.zeros(model.sample_size)))
        batched_reference = BatchTorchMesh(reference, 'reference', batch_size=2)

        random_walk = GaussianRandomWalkProposal(batched_reference.batch_size, np.zeros(model.sample_size))
        random_walk_2 = ClosestPointProposal(batched_reference.batch_size, np.zeros(model.sample_size), reference,
                                             target, model)
        random_walk_2.calculate_posterior_model()
        sampler = PDMMetropolisSampler(model, random_walk, batched_reference, target, correspondences=False)
        generator = np.random.default_rng()
        for i in range(10001):
            random = generator.random()
            if random < 0.6:
                proposal = ParameterProposalType.MODEL
            elif 0.6 <= random < 0.8:
                proposal = ParameterProposalType.TRANSLATION
            else:
                proposal = ParameterProposalType.ROTATION
            sampler.propose(proposal)
            sampler.determine_quality(proposal)
            sampler.decide()

        return batched_reference, model, sampler

    else:
        meshes = custom_io.read_meshes(mesh_path)

        # Currently, already registered meshes are required
        # if not registered:
        #     ref, landmarks = custom_io.read_landmarks_with_reference(reference_lm_path, landmark_path)
        #
        #     landmark_aligner = ProcrustesAnalyser(landmarks, ref)
        #     transforms, _, identifiers = landmark_aligner.generalised_procrustes_alignment()
        #
        #     dict_transforms = {identifier: transform for transform, identifier in zip(transforms, identifiers)}
        #     for i, mesh in enumerate(meshes):
        #         if mesh.id in dict_transforms:
        #             mesh.apply_transformation(dict_transforms.get(mesh.id))
        #             meshes[i] = mesh.simplify_qem(target=200)
        #         else:
        #             warnings.warn("Warning: Unexpected error occurred while transforming the meshes.",
        #                           UserWarning)
        #
        #     meshes = [mesh for mesh in meshes if mesh.id != 'reference']
        #     icp_aligner = ICPAnalyser(meshes)
        #     icp_aligner.icp()

        # Unnecessary if meshes are not modified
        # if write_meshes:
        #     write_path = Path().cwd().parent / Path(mesh_path).parent / 'meshes-prepared-2'
        #     write_path.mkdir(parents=True, exist_ok=True)
        #     for mesh in meshes:
        #         file_name = str(mesh.id) + '.stl'
        #         mesh_io = MeshReaderWriter(write_path / file_name)
        #         mesh_io.write_mesh(mesh)

        model = PointDistributionModel(meshes)

        if simplify_model:
            reference = model.decimate(200)
        else:
            reference = meshes[0]

        # TODO: Think again: Does it make sense to represent the target as a TorchMesh?
        target = reference.copy()
        target.set_points(model.get_points_from_parameters(1.0 * np.ones(model.sample_size)))
        # target = create_artificial_partial_target(target)
        reference.set_points(model.get_points_from_parameters(np.zeros(model.sample_size)))
        batched_reference = BatchTorchMesh(reference, 'reference', batch_size=2)

        random_walk = GaussianRandomWalkProposal(batched_reference.batch_size, np.zeros(model.sample_size))
        random_walk_2 = ClosestPointProposal(batched_reference.batch_size, np.zeros(model.sample_size), reference,
                                             target, model)
        random_walk_2.calculate_posterior_model()
        sampler = PDMMetropolisSampler(model, random_walk, batched_reference, target, correspondences=False)
        generator = np.random.default_rng()
        for i in range(40001):
            random = generator.random()
            if random < 0.6:
                proposal = ParameterProposalType.MODEL
            elif 0.6 <= random < 0.8:
                proposal = ParameterProposalType.TRANSLATION
            else:
                proposal = ParameterProposalType.ROTATION
            sampler.propose(proposal)
            sampler.determine_quality(proposal)
            sampler.decide()

        return batched_reference, model, sampler


class Main:
    def __init__(self):
        pass


if __name__ == "__main__":
    main = Main()
    # batched_reference, model, sampler = run("datasets/femur-data/project-data/registered",
    #                                         landmark_path="datasets/femur-data/project-data/landmarks",
    #                                         reference_lm_path="datasets/femur-data/project-data/reference-landmarks",
    #                                         reference_path="datasets/femur-data/project-data/reference-decimated",
    #                                         model_path="datasets/models",
    #                                         read_model=True,
    #                                         simplify_model=True,
    #                                         registered=True,
    #                                         write_meshes=False
    #                                         )
    batched_reference, model, sampler = run("datasets/femur-data/project-data/registered",
                                            model_path="datasets/models",
                                            reference_path="datasets/femur-data/project-data/reference-decimated",
                                            read_model=True,
                                            simplify_model=True
                                            )
    visualizer = MainVisualizer(batched_reference, model, sampler)
    print(sampler.acceptance_ratio())
    visualizer.run()
