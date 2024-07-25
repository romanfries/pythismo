import numpy as np
import torch
import trimesh.registration
from meshio import Mesh
from scipy.optimize import linear_sum_assignment

from src.mesh.TMesh import TorchMesh


def extract_reference(meshes):
    new_meshes = []
    for mesh in meshes:
        if mesh.id == 'reference':
            reference = mesh
        else:
            new_meshes.append(mesh)

    return new_meshes, reference


def distance_to_closest_point(ref_points, target_points, batch_size):
    # TODO: Adapt for two-dimensional input tensors (batch size equal to 1)
    num_target_points = target_points.shape[0]
    target_points_expanded = target_points.unsqueeze(2).expand(-1, -1, batch_size).unsqueeze(
        0)
    points_expanded = ref_points.unsqueeze(1)
    distances = torch.sub(points_expanded, target_points_expanded)
    closest_points = torch.argmin(torch.sum(torch.pow(distances, float(2)), dim=2), dim=0)
    distances = distances[closest_points,
                torch.arange(num_target_points).unsqueeze(1).expand(num_target_points, batch_size), :,
                torch.arange(batch_size)]
    return torch.transpose(distances, 1, 2)


class ICPAnalyser:
    def __init__(self, meshes, iterations=100):
        self.meshes = meshes[1:]
        self.reference = meshes[0]
        # self.meshes, self.reference = extract_reference(meshes)
        # This variant of ICP assumes that the initial transformation has already been applied to the meshes.
        self.transforms = None
        self.iterations = iterations

    def icp(self):
        reference_points = np.asanyarray(self.reference.points, dtype=np.float64)
        if not trimesh.util.is_shape(reference_points, (-1, 3)):
            raise ValueError("Invalid point shape.")
        for i, mesh in enumerate(self.meshes):
            _, transformed_points, _ = trimesh.registration.icp(mesh.points, self.reference.points,
                                                                max_iterations=self.iterations)
            mesh.set_points(transformed_points)
            # differences = distance_to_closest_point(mesh.tensor_points.unsqueeze(2), self.reference.tensor_points, 1) \
            #    .squeeze(-1)
            # mesh.set_points(self.reference.tensor_points + differences)
            distances = np.linalg.norm(transformed_points[:, np.newaxis, :] - reference_points[np.newaxis, :, :],
                                       axis=2)
            distances = np.power(distances, 2)
            row, col = linear_sum_assignment(distances)
            reordered_points = np.empty_like(mesh.points)
            reordered_points[col] = transformed_points[row]
            mesh.cells[0].data = col[mesh.cells[0].data]
            mesh.set_points(reordered_points)
