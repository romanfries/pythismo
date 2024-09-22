import numpy as np
import trimesh.registration


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


def calculate_mean_shape(landmarks_set):
    mean_shape = (1 / len(landmarks_set)) * np.mean(np.stack(landmarks_set), axis=0)
    return mean_shape


def align_to_reference(reference, landmarks_set):
    landmarks_set_aligned = []
    transforms = []
    for landmarks in landmarks_set:
        transform, landmarks_new, _ = trimesh.registration.procrustes(landmarks, reference, reflection=False,
                                                                      scale=False)
        transforms.append(transform)
        landmarks_set_aligned.append(landmarks_new)

    return transforms, landmarks_set_aligned


class ProcrustesAnalyser:
    def __init__(self, landmarks_set, reference=None, iterations=20, data_format='landmark_json'):
        if data_format == 'landmark_json':
            # landmarks_set is a list of tuples with the respective coordinates of the landmarks and the identifier that
            # clarifies to which mesh these landmarks belong.
            self.reference, self.landmarks_set, self.identifiers = prepare_landmark_json(reference, landmarks_set)
        elif data_format == 'mesh':
            # Important: Only functions as desired if correspondences are given and should therefore be used with
            # caution!
            self.landmarks_set = []
            self.identifiers = []
            for mesh in landmarks_set:
                self.landmarks_set.append(mesh.tensor_points.cpu())
                self.identifiers.append(mesh.id)
            self.reference = self.landmarks_set[0]
        else:
            raise NotImplementedError('Procrustes analysis not implemented for this data format.')
        self.data_format = data_format
        self.iterations = iterations
        self.cardinality = len(landmarks_set)

    def generalised_procrustes_alignment(self):
        # Algorithm as outlined in the lecture notes of the course "Statistical Shape Modelling" by Marcel Lüthi
        # 1: Choose a shape as the reference shape \Gamma_R
        # The class is defined in such a way that a specific shape is already passed as a reference.
        # 2: Align all shapes \Gamma^{(1)}, ..., \Gamma^{(n)} using the Procrustes alignment of two point sets
        transforms = []
        for index in range(self.cardinality):
            # Dimensionality 3 was hardcoded here
            transforms.append(np.identity(4))
        new_transforms, new_landmarks = align_to_reference(self.reference, self.landmarks_set)
        transforms = [np.matmul(a, b) for a, b in zip(transforms, new_transforms)]
        # The initial reference is not used any further!

        # Reinsert the reference
        if self.data_format == 'landmark_list':
            transforms.append(np.identity(4))
            new_transforms.append(np.identity(4))
            new_landmarks.append(np.array(self.reference))
            self.identifiers.append('reference')

        # 3: Compute the mean shape \Gamma_{\mu} for the set of aligned shapes
        for j in range(self.iterations - 1):
            mean_shape = calculate_mean_shape(new_landmarks)

            # 4: Iterate using the mean shape as the new reference shape
            new_transforms, new_landmarks = align_to_reference(mean_shape, new_landmarks)
            transforms = [np.matmul(a, b) for a, b in zip(transforms, new_transforms)]
        # new_landmarks does not have the same format as the originally inputted landmarks
        return transforms, new_landmarks, self.identifiers
