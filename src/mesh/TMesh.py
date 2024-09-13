import numpy as np
import meshio
from trimesh import Trimesh

import pytorch3d
from meshio import Mesh
import torch
from quad_mesh_simplify import simplify_mesh
import warnings


def get_transformation_matrix(angles: torch.Tensor) -> torch.Tensor:
    """
    Generates suitable transformation matrices for given Euler angles (ZYX convention).

    :param angles: Torch tensor with (optional: a batch of) Euler coordinates in radians (shape (3,) or
    (3, batch_size)).
    :type angles: torch.Tensor
    :return: Torch tensor with (a batch of) transformation matrix/matrices (shape (4, 4) or (4, 4, batch_size)).
    :rtype: torch.Tensor
    """
    if angles.ndim == 1:
        angles = angles[:, None]

    cx, cy, cz = torch.cos(angles)
    sx, sy, sz = torch.sin(angles)

    batch_size = angles.size()[1]
    transformation_matrices = torch.zeros((4, 4, batch_size), device=angles.device, dtype=angles.dtype)

    transformation_matrices[0, 0, :] = cy * cz
    transformation_matrices[0, 1, :] = -cx * sz + cz * sx * sy
    transformation_matrices[0, 2, :] = cx * cz * sy + sx * sz
    transformation_matrices[1, 0, :] = cy * sz
    transformation_matrices[1, 1, :] = cx * cz + sx * sy * sz
    transformation_matrices[1, 2, :] = cx * sy * sz - cz * sx
    transformation_matrices[2, 0, :] = -sy
    transformation_matrices[2, 1, :] = cy * sx
    transformation_matrices[2, 2, :] = cx * cy
    transformation_matrices[3, 3, :] = 1

    if batch_size == 1:
        transformation_matrices = transformation_matrices.squeeze(-1)

    return transformation_matrices


class TorchMeshGpu(Mesh):
    def __init__(self, mesh, identifier, dev):
        """
        This class is an extension of the meshio.Mesh class. It extends the superclass with a torch.Tensor to perform
        mesh calculations using PyTorch. The dimensionality is fixed at 3.
        For more information on meshio, please visit https://github.com/nschloe/meshio.

        :param mesh: Mesh instance that serves as a base.
        :type mesh: meshio.Mesh
        :param identifier: Identification string.
        :type identifier: str
        :param dev: An object representing the device on which the tensor operations are or will be allocated.
        :type dev: torch.device
        """
        # possibly prone to errors
        arg_dict = vars(mesh)
        arg_dict = {key: value for key, value in arg_dict.items() if key not in {'tensor_points', 'id', 'num_points',
                                                                                 'dimensionality', 'initial_com',
                                                                                 'dev', 'calculated_facet_normals'}}
        args = list(arg_dict.values())
        super().__init__(*args)
        self.dev = dev
        # If the condition is met, the constructor was called from the ‘BatchTorchMesh’ subclass. Therefore, the warning
        # can be ignored.
        if isinstance(self.points, np.ndarray) and np.array_equal(self.points, np.array(None, dtype=object)):
            self.tensor_points = mesh.tensor_points
        else:
            self.tensor_points = torch.tensor(self.points, dtype=torch.float32, device=self.dev)
        self.points = None
        # Avoid constructing a new Torch tensor from an existing tensor.
        if isinstance(self.cells[0].data, torch.Tensor):
            self.cells[0].data.to(self.dev)
        else:
            self.cells[0].data = torch.tensor(self.cells[0].data, device=self.dev)
        self.num_points = self.tensor_points.size()[0]
        self.dimensionality = self.tensor_points.size()[1]
        self.id = identifier
        self.initial_com = self.center_of_mass()
        self.calculated_facet_normals = False

    @property
    def cells_dict(self):
        """
        Adaptation of an internal meshio.Mesh method to ensure compatibility with Torch.

        :return: Dictionary containing the triangle data.
        :rtype: dict
        """
        cells_dict = {}
        for cell_block in self.cells:
            if cell_block.type not in cells_dict:
                cells_dict[cell_block.type] = []
            cells_dict[cell_block.type].append(cell_block.data)
        # concatenate
        for key, value in cells_dict.items():
            cells_dict[key] = value
        return cells_dict

    @property
    def cell_data_dict(self):
        """
        Adaptation of an internal meshio.Mesh method to ensure compatibility with Torch.

        :return: Dictionary containing a dictionary with the added additional triangle data.
        :rtype: dict
        """
        cell_data_dict = {}
        for key, value_list in self.cell_data.items():
            cell_data_dict[key] = {}
            for value, cell_block in zip(value_list, self.cells):
                if cell_block.type not in cell_data_dict[key]:
                    cell_data_dict[key][cell_block.type] = []
                cell_data_dict[key][cell_block.type].append(value)

            for cell_type, val in cell_data_dict[key].items():
                cell_data_dict[key][cell_type] = val
        return cell_data_dict

    @classmethod
    def from_mesh(cls, mesh, identifier, dev):
        """
        Redundant method.

        :param mesh: Mesh instance that serves as a base.
        :type mesh: meshio.Mesh
        :param identifier: Identification string.
        :type identifier: str
        :param dev: An object representing the device on which the tensor operations are or will be allocated.
        :type dev: torch.device
        :return Newly created TorchMeshGpu instance.
        :rtype: TorchMeshGpu
        """
        return cls(mesh, identifier, dev)

    def simplify_qem(self, target: int = 1000):
        """
        Reduces the mesh to the specified number of points and returns a new instance with these new points/triangles.
        The identifier remains unchanged.
        The algorithm used comes from the paper by M. Garland and P. Heckbert: "Surface Simplification Using Quadric
        Error Metrics" (1997) implemented by J. Magnusson. For more information, please visit
        https://github.com/jannessm/quadric-mesh-simplification.
        Remark: The function simplify_mesh and the library meshio still expect NumPy arrays, so the PyTorch tensors need
        to be converted to NumPy arrays before passing them to these functions, and then the results are converted back
        to PyTorch tensors after processing. This can be disadvantageous for the runtime.

        :param target: Number of nodes that the final mesh will have.
        :type target: int
        :return: Reduced TorchMesh instance with the specified number of points/nodes.
        :rtype: TorchMeshGpu
        """
        new_coordinates, new_triangles = simplify_mesh(self.tensor_points.cpu().numpy().astype(np.float64),
                                                       self.cells[0].data.cpu().numpy().astype(np.uint32),
                                                       num_nodes=target)

        new_mesh = meshio.Mesh(new_coordinates.astype(np.float32),
                               [meshio.CellBlock('triangle', new_triangles.astype(np.int64))])
        return self.from_mesh(new_mesh, self.id, self.dev)

    def set_points(self, transformed_points, reset_com=False):
        """
        The method for changing the points of the mesh. The cells (triangles) remain unchanged.

        :param transformed_points: The new coordinates of the points. The shape of the
        Torch tensor has to be (num_points, 3).
        :type transformed_points: torch.Tensor
        :param reset_com: Boolean value that specifies whether the center of mass of the mesh should be recalculated.
        :type reset_com: bool
        """
        # transformed_points: Torch tensor
        self.tensor_points = transformed_points
        if reset_com:
            self.initial_com = self.center_of_mass()

    def apply_transformation(self, transform: torch.Tensor):
        """
        Changes the points of the mesh by multiplying them by a (4, 4) transformation matrix.

        :param transform: Transformation matrix with shape (4, 4).
        :type transform: torch.Tensor
        :param save_old: Boolean value that specifies whether the old point coordinates should be saved. Note: This is
        only relevant if the method is called from a BatchTorchMesh instance. TorchMesh does not offer the option of
        saving the old coordinates.
        :type save_old: bool
        """
        additional_col = torch.ones((self.tensor_points.size()[0], 1), device=self.dev)
        extended_points = torch.cat((self.tensor_points, additional_col), dim=1)
        transformed_points = extended_points @ transform.T
        transformed_points = transformed_points[:, :3]
        self.set_points(transformed_points)

    def apply_translation(self, translation_parameters, save_old=False):
        """
        Changes the points of the mesh by adding a (3,) translation vector.

        :param translation_parameters: Tensor with shape (3,) containing the translation parameters.
        :type translation_parameters: torch.Tensor
        :param save_old: Boolean value that specifies whether the old point coordinates should be saved. Note: This is
        only relevant if the method is called from a BatchTorchMesh instance. TorchMesh does not offer the option of
        saving the old coordinates.
        :type save_old: bool
        """
        translated_points = self.tensor_points + translation_parameters
        self.set_points(translated_points)

    def apply_rotation(self, rotation_parameters, save_old=False):
        """
        Rotates the mesh instance using given Euler angles (ZYX convention).

        :param rotation_parameters: Torch tensor with shape (3,) / (3, batch_size) [when called from a BatchTorchMesh
        instance] containing the Euler angles (in radians).
        :type rotation_parameters: torch.Tensor
        :param save_old: Boolean value that specifies whether the old point coordinates should be saved. Note: This is
        only relevant if the method is called from a BatchTorchMesh instance. TorchMeshGpu does not offer the option of
        saving the old coordinates.
        :type save_old: bool
        """
        transformation = get_transformation_matrix(rotation_parameters)
        self.apply_translation(- self.initial_com, save_old)
        self.apply_transformation(transformation)
        self.apply_translation(self.initial_com)

    def center_of_mass(self):
        """
        Calculates the centre of mass of the mesh object.

        :return: Coordinates of the centre of mass with shape (3,).
        :rtype: torch.Tensor
        """
        return torch.sum(self.tensor_points, dim=0) / self.num_points

    def to_pytorch3d_pointclouds(self, batch_size=1):
        """
        Helper method to translate a given TorchMeshGpu into a pytorch3d.structures.Pointclouds object.
        The Pointclouds class provides functions for working with batches of 3d point clouds, and converting between
        representations.
        The generated Pointclouds object represents ‘batch_size’ times the same point cloud, namely the one consisting
        of the points of the calling TorchMeshGpu instance.

        :param batch_size: Specifies how often the point cloud of the TorchMeshGpu is repeated in the Pointclouds
        object.
        :type batch_size: int
        :return: Above described Pointclouds object.
        :rtype: pytorch3d.structures.Pointclouds
        """
        points = self.tensor_points.unsqueeze(0).expand(batch_size, -1, -1)
        return pytorch3d.structures.Pointclouds(points)

    def to_pytorch3d_meshes(self, batch_size=1):
        """
        Helper method to translate a given TorchMeshGpu into a pytorch3d.structures.Meshes object.
        The Meshes class provides functions for working with batches of triangulated meshes with varying numbers of
        faces and vertices, and converting between representations.
        However, the generated Meshes object here is more restricted, because the meshes in the BatchTorchMesh instance
        are all exactly the same.

        :return: Above described Meshes object.
        :rtype: pytorch3d.structures.Meshes
        """
        verts = self.tensor_points.unsqueeze(0).expand(batch_size, -1, -1)
        faces = self.cells[0].data.unsqueeze(0).expand(batch_size, -1, -1)
        return pytorch3d.structures.Meshes(verts, faces)

    def calc_facet_normals(self):
        """
        (Re-)Calculates the unit-length normal vectors to the triangular surfaces of the mesh.
        """
        triangles = self.cells[0].data
        v0, v1, v2 = self.tensor_points[triangles].unbind(dim=1)
        edges_a, edges_b = v1 - v0, v2 - v0
        facet_normals = torch.linalg.cross(edges_a, edges_b)
        facet_normals = facet_normals / torch.norm(facet_normals, dim=1, keepdim=True)
        self.calculated_facet_normals = True
        self.cell_data.update({"facet_normals": [facet_normals]})

    def change_device(self, dev):
        """
        Change the device on which the tensor operations are or will be allocated.

        :param dev: The future device on which the mesh data is to be saved.
        :type dev: torch.device
        """
        if self.dev == dev:
            return
        else:
            self.tensor_points = self.tensor_points.to(dev)
            self.cells[0].data = self.cells[0].data.to(dev)
            self.initial_com = self.initial_com.to(dev)
            if self.calculated_facet_normals:
                self.cell_data["facet_normals"][0].data = self.cell_data["facet_normals"][0].data.to(dev)

            self.dev = dev

    def partial_shape(self, idx1, idx2, ratio_observed):
        """
        Takes the two indices of the points that define the length of the mesh and the proportion of the desired
        observed part and determines the corresponding partial mesh.

        :param idx1: Index of the highest point of the mesh.
        :type idx1: int
        :param idx2: Index of the lowest point of the mesh.
        :type idx2: int
        :param ratio_observed: Proportion of the desired observed part.
        :type ratio_observed: float
        :return: Partial shape as a TorchMeshGpu instance.
        :rtype: TorchMeshGpu
        """
        plane_normal = (self.tensor_points[idx1, :] - self.tensor_points[idx2, :])
        plane_origin = (self.tensor_points[idx2, :] + ratio_observed * plane_normal)
        mesh_tri = Trimesh(self.tensor_points.cpu(), self.cells[0].data.cpu()).slice_plane(plane_origin.cpu(),
                                                                                           plane_normal.cpu())
        return TorchMeshGpu(
            meshio.Mesh(np.array((mesh_tri.vertices), [meshio.CellBlock('triangle', np.array(mesh_tri.faces))]),
            'partial_shape', self.dev)


class BatchTorchMesh(TorchMeshGpu):
    def __init__(self, mesh, identifier, dev, batch_size=50, batched_data=False, batched_points=None):
        """
        Further extension of the TorchMesh class. Saves 'batch_size' many mesh instances simultaneously.
        However, these meshes differ only in the point coordinates and all have the same triangulation.
        There are two types of instantiation. Either only a TorchMesh is given, whose point coordinates are then
        replicated accordingly, or a batch of point coordinates.

        :param mesh: TorchMesh instance, which serves as the basis. The triangle data is transferred from it.
        :type mesh: TorchMeshGpu
        :param identifier: Identification string.
        :type identifier: str
        :param dev: An object representing the device on which the tensor operations are or will be allocated.
        :type dev: torch.device
        :param batch_size: Self-explanatory.
        :type batch_size: int
        :param batched_data: Boolean variable that specifies whether the BatchTorchMesh is initialized with a batch of
        (different) point coordinates.
        :type batched_data: bool
        :param batched_points: 3-dimensional tensor with shape (num_points, dimensionality, batch_size) with the batched
        point coordinates.
        :type batched_points: torch.Tensor
        """
        # possibly prone to errors
        super().__init__(mesh, identifier, dev)
        if not batched_data:
            self.tensor_points = self.tensor_points.unsqueeze(2).expand(-1, -1, batch_size)
            self.batch_size = batch_size
        else:
            self.tensor_points = batched_points
            self.tensor_points.to(self.dev)
            self.batch_size = self.tensor_points.size()[2]
        self.old_points = None
        self.initial_com = self.center_of_mass()

    @classmethod
    def from_mesh(cls, mesh, identifier, dev, batch_size=50):
        """
        Redundant method.

        :param mesh: TorchMesh instance, which serves as the basis. The triangle data is transferred from it.
        :type mesh: TorchMeshGpu
        :param identifier: Identification string.
        :type identifier: str
        :param dev: An object representing the device on which the tensor operations are or will be allocated.
        :type dev: torch.device
        :param batch_size: Self-explanatory.
        :type batch_size: int
        :return Newly created BatchTorchMesh instance.
        :rtype: BatchTorchMesh
        """
        return cls(mesh, identifier, dev, batch_size=batch_size)

    def simplify_qem(self, target=1000):
        """
        This method should not be used for BatchTorchMesh instances.

        :param target:
        """
        warnings.warn("Warning: TorchMeshGpu method invoked from BatchTorchMesh instance. No action taken.",
                      UserWarning)

    def set_points(self, transformed_points, save_old=False, reset_com=False):
        """
        The method for changing the points of the mesh. The cells (triangles) remain unchanged.

        :param transformed_points: The new coordinates of the points. The shape of the Torch tensor has to be
        (num_points, 3, batch_size).
        :type transformed_points: torch.Tensor
        :param save_old: Boolean value that specifies whether the old point coordinates should be saved.
        :type save_old: bool
        :param reset_com: Boolean value that specifies whether the center of mass of the mesh should be recalculated.
        :type reset_com: bool
        """
        if save_old:
            self.old_points = self.tensor_points.clone()
        else:
            self.old_points = None
        self.tensor_points = transformed_points
        if reset_com:
            self.initial_com = self.center_of_mass()

    def apply_transformation(self, transform, save_old=False):
        """
        Changes the points of the mesh by multiplying them by a (4, 4) transformation matrix.
        A separate transformation matrix must be provided for each element of the batch.
        Bear this in mind if the same transformation is to be applied to all elements of the batch by extending the
        'transform' array accordingly.

        :param transform: Transformation matrix with shape (4, 4, batch_size).
        :type transform: torch.Tensor
        :param save_old: Boolean value that specifies whether the old point coordinates should be saved.
        :type save_old: bool
        """
        additional_cols = torch.ones((self.tensor_points.size()[0], 1, self.tensor_points.size()[2]),
                                     device=self.tensor_points.device)
        extended_points = torch.cat((self.tensor_points, additional_cols), dim=1)
        transformed_points = torch.einsum('ijk,ljk->lik', transform, extended_points)
        transformed_points = transformed_points[:, :3, :]

        if save_old:
            self.old_points = self.tensor_points.clone()
        else:
            self.old_points = None
        self.set_points(transformed_points, save_old)

    def apply_translation(self, translation_parameters, save_old=False):
        """
        Changes the points of the mesh by adding a (3,) translation vector.
        A separate translation vector must be provided for each element of the batch.
        Bear this in mind if the same translation is to be applied to all elements of the batch by extending the
        'translation_parameters' array accordingly.

        :param translation_parameters: Torch tensor with shape (3, batch_size) containing the translation parameters.
        :type translation_parameters: torch.Tensor
        :param save_old: Boolean value that specifies whether the old point coordinates should be saved.
        :type save_old: bool
        """
        translated_points = self.tensor_points + translation_parameters
        if save_old:
            self.old_points = self.tensor_points
        else:
            self.old_points = None
        self.set_points(translated_points, save_old)

    def update_points(self, decider):
        """
        Internal method that updates the point values according to the information from the passed decider.

        :param decider: Boolean tensor of the shape (batch_size,), which indicates for each element of the batch whether
        the new point values were accepted (=True) or rejected (=False).
        :type decider: torch.Tensor
        """
        self.tensor_points = torch.where(decider.unsqueeze(0).unsqueeze(1), self.tensor_points,
                                         self.old_points)
        self.old_points = None

    def to_pytorch3d_pointclouds(self, batch_size=1):
        """
        Helper method to translate a given BatchTorchMesh into a pytorch3d.structures.Pointclouds object.
        The Pointclouds class provides functions for working with batches of 3d point clouds, and converting between
        representations.

        :return: Above described Pointclouds object.
        :rtype: pytorch3d.structures.Pointclouds
        """
        points = torch.permute(self.tensor_points, (2, 0, 1))
        return pytorch3d.structures.Pointclouds(points)

    def to_pytorch3d_meshes(self, batch_size=1):
        """
        Helper method to translate a given BatchTorchMesh into a pytorch3d.structures.Meshes object.
        The Meshes class provides functions for working with batches of triangulated meshes with varying numbers of
        faces and vertices, and converting between representations.
        However, the generated Meshes object here is more restricted, because the meshes in the BatchTorchMesh instance
        all have the same number of points and the exact same triangulation.

        :return: Above described Meshes object.
        :rtype: pytorch3d.structures.Meshes
        """
        verts = torch.permute(self.tensor_points, (2, 0, 1))
        faces = self.cells[0].data.unsqueeze(0).expand(self.batch_size, -1, -1)
        return pytorch3d.structures.Meshes(verts, faces)

    def calc_facet_normals(self):
        """
        (Re-)Calculates the unit-length normal vectors to the triangular surfaces of the mesh.
        """
        triangles = self.cells[0].data
        v0, v1, v2 = self.tensor_points[triangles].unbind(dim=1)
        edges_a, edges_b = v1 - v0, v2 - v0
        facet_normals = torch.linalg.cross(edges_a, edges_b, dim=1)
        facet_normals = facet_normals / torch.norm(facet_normals, dim=1, keepdim=True)
        self.calculated_facet_normals = True
        self.cell_data.update({"facet_normals": [facet_normals]})

    def partial_shape(self, idx1, idx2, ratio_observed):
        """
        This method should not be used for BatchTorchMesh instances.

        :param idx1:
        :param idx2:
        :param ratio_observed:
        """
        warnings.warn("Warning: TorchMeshGpu method invoked from BatchTorchMesh instance. No action taken.",
                      UserWarning)
