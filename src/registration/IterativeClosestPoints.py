import numpy as np
import trimesh.registration
from scipy.optimize import linear_sum_assignment


def extract_reference(meshes):
    new_meshes = []
    for mesh in meshes:
        if mesh.id == 'reference':
            reference = mesh
        else:
            new_meshes.append(mesh)

    return new_meshes, reference


class ICPAnalyser:
    def __init__(self, meshes, transforms=None, iterations=20):
        self.meshes, self.reference = extract_reference(meshes)
        self.transforms = None
        self.iterations = iterations

    def icp(self):
        reference_points = np.asanyarray(self.reference.points, dtype=np.float64)
        if not trimesh.util.is_shape(reference_points, (-1, 3)):
            raise ValueError("Invalid point shape.")
        for mesh in self.meshes:
            _, transformed_points, _ = trimesh.registration.icp(mesh.points, self.reference.points)
            mesh.set_points(transformed_points)
            distances = np.linalg.norm(transformed_points[:, np.newaxis, :] - reference_points[np.newaxis, :, :], axis=2)
            distances = np.power(distances, 2)
            row, col = linear_sum_assignment(distances)
            reordered_points = np.empty_like(mesh.points)
            reordered_points[col] = transformed_points[row]
            mesh.cells[0].data = col[mesh.cells[0].data]
            mesh.set_points(reordered_points)









