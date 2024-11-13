from enum import Enum

import pytorch3d
import torch


def extract_reference(meshes, identifier):
    """
    NOTE: Method is currently not used.
    Method for extracting a mesh with a specific identifier 'identifier' from a list of meshes.
    This mesh is removed from the list.

    :param meshes: List of mesh instances to be searched.
    :type meshes: list of TorchMeshGpu
    :param identifier: Identifier string to be searched for.
    :type identifier: str
    :return: Tuple with two elements:
        - list of TorchMeshGpu: Original list of meshes without the mesh extracted.
        - TorchMeshGpu: Extracted mesh instance.
    """
    new_meshes = []
    reference = None
    for mesh in meshes:
        if mesh.id == identifier:
            reference = mesh
        else:
            new_meshes.append(mesh)

    return new_meshes, reference


def difference_to_closest_point(ref_points, target_points, batch_size):
    # TODO: Adapt for two-dimensional input tensors (batch size equal to 1)
    """
    NOTE: Method is currently not used.
    Calculates the negative difference between each point of a target and the point of a reference closest to this
    target point. An entire batch of references is analysed this way.

    :param ref_points: Coordinates of the points of the references as a 3-dimensional tensor with shape
    (num_points_ref, dimensionality, batch_size).
    :type ref_points: torch.Tensor
    :param target_points: Coordinates of the points of the target as a 2-dimensional tensor with shape:
    (num_points_target, dimensionality).
    :type target_points: torch.Tensor
    :param batch_size: Specifies how many references are analysed simultaneously.
    :type batch_size: int
    :return: Tensor containing negative differences between each point of a target and the point of a reference closest
    to this target point with shape (num_points_target, dimensionality, batch_size).
    :rtype: torch.Tensor
    """
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


class ICPMode(Enum):
    SINGLE = 1
    BATCHED = 2


class ICPAnalyser:
    def __init__(self, meshes, targets=None, iterations=100, mode=ICPMode.SINGLE):
        """
        Class to rigidly align a set of point clouds/meshes using the Iterative Closest Points (ICP) algorithm.
        If no targets are specified (SINGLE mode), all meshes are aligned to the first entry in the mesh list
        'meshes' that is selected as a reference.
        The second mode (BATCHED mode) expects 'targets' and 'meshes' to be a BatchTorchMesh of equal batch size.
        Each shape of the batch in 'meshes' is aligned independently to its corresponding target in 'targets'.
        Remark: ICP will only produce reasonable results if the shapes are already roughly aligned. This method does not
        use a parameter for an initial transformation.
        Remark 2: Naming conventions chosen as in the lecture notes of the course "Statistical Shape Modelling" by
        Marcel Lüthi.

        :param meshes: List or BatchTorchMesh of mesh instances to be aligned.
        :type meshes: list of TorchMeshGpu or BatchTorchMesh
        :param targets: Optional, BatchTorchMesh with individual targets for the elements of 'meshes'.
        :type targets: None or BatchTorchMesh
        :param iterations: Maximum number of iterations for the alignment.
        :type iterations: int
        :param mode: Defines whether the class operates in SINGLE or BATCHED mode.
        :type mode: ICPMode
        """
        self.mode = mode
        self.iterations = iterations

        if mode == ICPMode.SINGLE:
            self.reference = meshes[1:]
            self.target = meshes[0]
        elif mode == ICPMode.BATCHED:
            if targets is None:
                raise ValueError("Targets must not be None in BATCHED mode.")
            self.reference = meshes
            self.target = targets
        else:
            raise ValueError("Invalid mode. Use ICPMode.SINGLE or ICPMode.BATCHED.")
        self.transformation = None

    def icp(self, swap_transformation=False, calc_facet_normals=False):
        """
        Method that must be called to execute the alignment. After the call, `reference` will contain
        the transformed point clouds/meshes.
        """
        if self.mode == ICPMode.SINGLE:
            target_points = self.target.tensor_points
            reference_points_list = [mesh.tensor_points for mesh in self.reference]
            reference_points = torch.stack(reference_points_list)
            target_points = target_points.unsqueeze(0).expand(reference_points.size()[0], -1, -1)
            icp_solution = pytorch3d.ops.iterative_closest_point(target_points, reference_points,
                                                                 max_iterations=self.iterations)
            if not swap_transformation:
                transformed_points = torch.matmul(reference_points - icp_solution.RTs.T.unsqueeze(1),
                                                  torch.linalg.inv(icp_solution.RTs.R))
                if calc_facet_normals:
                    _ = list(map(lambda x, y: x.set_points(y, adjust_rotation_centre=True), self.reference, transformed_points))
                else:
                    _ = list(map(lambda x, y: x.set_points(y, adjust_rotation_centre=True, calc_facet_normals=False), self.reference, transformed_points))
                self.transformation = torch.linalg.inv(icp_solution.RTs.R), -(icp_solution.RTs.T.unsqueeze(-2) @ icp_solution.RTs.R).squeeze(1)
            else:
                # Make sure there is only a single mesh in the list reference!
                transformed_points = torch.matmul(target_points, icp_solution.RTs.R) + icp_solution.RTs.T.unsqueeze(1)
                if calc_facet_normals:
                    self.target.set_points(transformed_points.squeeze(), adjust_rotation_centre=True)
                else:
                    self.target.set_points(transformed_points.squeeze(), adjust_rotation_centre=True, calc_facet_normals=False)
                self.transformation = icp_solution.RTs.R, icp_solution.RTs.T
        elif self.mode == ICPMode.BATCHED:
            if self.target.tensor_points.shape[2] != self.reference.tensor_points.shape[2]:
                raise ValueError("In BATCHED mode, 'targets' and 'references' must have the same batch size.")

            reference_points = self.reference.tensor_points.permute(2, 0, 1)
            target_points = self.target.tensor_points.permute(2, 0, 1)
            icp_solution = pytorch3d.ops.iterative_closest_point(target_points, reference_points,
                                                                 max_iterations=self.iterations)
            if not swap_transformation:
                transformed_points = torch.matmul(reference_points - icp_solution.RTs.T.unsqueeze(1),
                                                  torch.linalg.inv(icp_solution.RTs.R))
                if calc_facet_normals:
                    self.reference.set_points(transformed_points.permute(1, 2, 0), adjust_rotation_centre=True)
                else:
                    self.reference.set_points(transformed_points.permute(1, 2, 0), adjust_rotation_centre=True, calc_facet_normals=False)
                self.transformation = torch.linalg.inv(icp_solution.RTs.R), -(icp_solution.RTs.T.unsqueeze(-2) @ icp_solution.RTs.R).squeeze(1)
            else:
                transformed_points = torch.matmul(target_points, icp_solution.RTs.R) + icp_solution.RTs.T.unsqueeze(1)
                if calc_facet_normals:
                    self.target.set_points(transformed_points.permute(1, 2, 0), adjust_rotation_centre=True)
                else:
                    self.target.set_points(transformed_points.permute(1, 2, 0), adjust_rotation_centre=True, calc_facet_normals=False)
                self.transformation = icp_solution.RTs.R, icp_solution.RTs.T
