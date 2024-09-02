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


class ICPAnalyser:
    def __init__(self, meshes, iterations=100):
        """
        Class to rigidly align a set of meshes using the Iterative Closest Points (ICP) algorithm.
        All meshes are aligned to the first entry in the mesh list 'meshes' that is selected as a reference.
        It is assumed that an initial transformation has already been applied to ensure that the meshes are
        approximately aligned. If this is not the case, the result may be poor because the algorithm has become trapped
        in a local minimum.

        :param meshes: List of mesh instances to be aligned.
        :type meshes: list of TorchMeshGpu
        :param iterations: Number of iterations per alignment
        :type iterations: int
        """
        self.meshes = meshes[1:]
        self.reference = meshes[0]
        # self.meshes, self.reference = extract_reference(meshes)
        # This variant of ICP assumes that the initial transformation has already been applied to the meshes.
        self.transforms = None
        self.iterations = iterations

    def icp(self):
        """
        Method that must be called in order to actually execute the alignment step. After the call, ‘meshes’ contains
        the transformed meshes.
        """
        reference_points = self.reference.tensor_points
        mesh_points_list = [mesh.tensor_points for mesh in self.meshes]
        mesh_points = torch.stack(mesh_points_list)
        reference_points = reference_points.unsqueeze(0).repeat(mesh_points.size()[0], 1, 1)
        icp_solution = pytorch3d.ops.iterative_closest_point(mesh_points, reference_points,
                                                             max_iterations=self.iterations)
        transformed_points = icp_solution.Xt
        _ = list(map(lambda x, y: x.set_points(y, reset_com=True), self.meshes, transformed_points))
