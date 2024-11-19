from enum import Enum

import torch
from operator import attrgetter

import pytorch3d.ops


def prepare_landmark_json(reference, landmarks_set):
    reference_list = []
    for reference_landmark in reference:
        reference_coordinates = reference_landmark.get('coordinates')
        reference_list.append(reference_coordinates)

    coordinates_list = []
    identifier_list = []
    for landmark_tuple in landmarks_set:
        landmarks, identifier = landmark_tuple
        coordinate_list = []
        for landmark in landmarks:
            landmark_coordinates = landmark.get('coordinates')
            coordinate_list.append(landmark_coordinates)
        coordinates_list.append(coordinate_list)
        identifier_list.append(identifier)
    return reference_list, coordinates_list, identifier_list


class PCAMode(Enum):
    SINGLE = 1
    BATCHED = 2


class ProcrustesAnalyser:
    def __init__(self, meshes, references=None, iterations=20, mode=PCAMode.SINGLE):
        # Important: Only functions as desired if fixed_correspondences are given and should therefore be used with
        # caution!
        self.mode = mode
        self.iterations = iterations

        self.meshes = meshes
        self.references = references
        if self.mode == PCAMode.SINGLE:
            self.points = torch.stack(list(map(attrgetter('tensor_points'), self.meshes)))
        elif self.mode == PCAMode.BATCHED:
            self.points = self.meshes.tensor_points.permute(2, 0, 1)
        self.transformation = None

    def generalised_procrustes_alignment(self, calc_facet_normals=False):
        if self.mode == PCAMode.SINGLE:
            # Algorithm as outlined in the lecture notes of the course "Statistical Shape Modelling" by Marcel Lüthi
            # 1: Choose a shape as the reference shape \Gamma_R
            reference = self.points[0, :, :]
            # 2: Align all shapes \Gamma^{(1)}, ..., \Gamma^{(n)} using the Procrustes alignment of two point sets
            reference = reference.repeat(self.points.size(0), 1, 1)
            rotations, translations, _ = pytorch3d.ops.corresponding_points_alignment(self.points, reference)
            self.points = self.points @ rotations + translations.unsqueeze(1)
            self.transformation = rotations, translations
            # The initial reference is not used any further!

            # 3: Compute the mean shape \Gamma_{\mu} for the set of aligned shapes
            for _ in range(self.iterations):
                mean = torch.mean(self.points, dim=0)

                # 4: Iterate using the mean shape as the new reference shape
                reference = mean.repeat(self.points.size(0), 1, 1)
                rotations, translations, _ = pytorch3d.ops.corresponding_points_alignment(self.points, reference)
                self.points = self.points @ rotations + translations.unsqueeze(1)
                self.transformation = self.transformation[0] @ rotations, (
                            self.transformation[1].unsqueeze(1) @ rotations + translations.unsqueeze(1)).squeeze()
            # Save the changes in the original meshes
            if calc_facet_normals:
                _ = list(map(lambda x, y: x.set_points(y, adjust_rotation_centre=True), self.meshes, self.points))
            else:
                _ = list(
                    map(lambda x, y: x.set_points(y, adjust_rotation_centre=True, calc_facet_normals=False), self.meshes,
                        self.points))

    def procrustes_alignment(self):
        if self.mode == PCAMode.BATCHED:
            if self.meshes.tensor_points.shape[2] != self.references.tensor_points.shape[2]:
                raise ValueError("In BATCHED mode, 'targets' and 'references' must have the same batch size.")

            references_points = self.references.tensor_points.permute(2, 0, 1)
            rotations, translations, _ = pytorch3d.ops.corresponding_points_alignment(self.points, references_points)
            return rotations, translations



