from pathlib import Path

import os

import numpy as np
import torch

import custom_io
from model.PointDistribution import PointDistributionModel
from src.sampling.Metropolis import PDMMetropolisSampler
from src.custom_io.H5ModelIO import ModelReader
from src.mesh.TMesh import BatchTorchMesh, create_artificial_partial_target
from src.sampling.proposals.ClosestPoint import ClosestPointProposal
from src.sampling.proposals.GaussRandWalk import GaussianRandomWalkProposal, ParameterProposalType
from visualization.DashViewer import MainVisualizer

# Important notes for running Pytorch3d on Windows:
# https://stackoverflow.com/questions/62304087/installing-pytorch3d-fails-with-anaconda-and-pip-on-windows-10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run(dev,
        mesh_path=None,
        reference_path=None,
        model_path=None,
        read_model=False,
        simplify_model=False):
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    if read_model:
        rel_model_path = Path(model_path)
        model_path = Path.cwd().parent / rel_model_path
        model_reader = ModelReader(model_path)
        reference = custom_io.read_meshes(reference_path, dev)[0]
        model = model_reader.get_model(dev)
        # TODO: Think again: Does it make sense to represent the target as a TorchMeshGpu?
        target = reference.copy()
        target.set_points(model.get_points_from_parameters(1.0 * torch.ones(model.rank, device=dev)), reset_com=True)
        target = create_artificial_partial_target(target)
        reference.set_points(model.get_points_from_parameters(torch.zeros(model.rank, device=dev)), reset_com=True)
        batched_reference = BatchTorchMesh(reference, 'reference', dev, batch_size=2)

        random_walk = GaussianRandomWalkProposal(batched_reference.batch_size, torch.zeros(model.rank, device=dev), dev)
        random_walk_2 = ClosestPointProposal(batched_reference.batch_size, torch.zeros(model.rank, device=dev), dev,
                                             reference, batched_reference, target, model)
        sampler = PDMMetropolisSampler(model, random_walk_2, batched_reference, target, correspondences=False)
        generator = torch.Generator(device=dev)
        for i in range(1001):
            random = torch.rand(1, device=dev, generator=generator).item()
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
        meshes = custom_io.read_meshes(mesh_path, dev)

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

        model = PointDistributionModel(meshes=meshes)

        if simplify_model:
            reference = model.decimate(1000)
        else:
            reference = meshes[0]

        # TODO: Think again: Does it make sense to represent the target as a TorchMeshGpu?
        target = reference.copy()
        target.set_points(model.get_points_from_parameters(1.0 * torch.ones(model.rank, device=dev)), reset_com=True)
        target = create_artificial_partial_target(target)
        reference.set_points(model.get_points_from_parameters(torch.zeros(model.rank, device=dev)), reset_com=True)
        batched_reference = BatchTorchMesh(reference, 'reference', dev, batch_size=2)

        random_walk = GaussianRandomWalkProposal(batched_reference.batch_size, torch.zeros(model.rank, device=dev), dev)
        random_walk_2 = ClosestPointProposal(batched_reference.batch_size, torch.zeros(model.rank, device=dev), dev,
                                             reference, batched_reference, target, model)
        sampler = PDMMetropolisSampler(model, random_walk_2, batched_reference, target, correspondences=False)
        generator = torch.Generator(device=dev)
        for i in range(1):
            random = torch.rand(1, device=dev, generator=generator).item()
            if random < 0.6:
                proposal = ParameterProposalType.MODEL
            elif 0.6 <= random < 0.8:
                proposal = ParameterProposalType.TRANSLATION
            else:
                proposal = ParameterProposalType.ROTATION
            sampler.propose(proposal)
            sampler.determine_quality(proposal)
            sampler.decide()

        return batched_reference, random_walk_2.posterior_model, sampler


class Main:
    def __init__(self):
        self.device = device


if __name__ == "__main__":
    # torch.set_default_tensor_type(torch.cuda.FloatTensor)
    main = Main()
    dev = main.device
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
    batched_reference, model, sampler = run(dev,
                                            "datasets/femur-data/project-data/registered",
                                            model_path="datasets/models",
                                            reference_path="datasets/femur-data/project-data/reference-decimated",
                                            read_model=False,
                                            simplify_model=False
                                            )
    batched_reference.change_device(torch.device("cpu"))
    model.change_device(torch.device("cpu"))
    sampler.change_device(torch.device("cpu"))
    visualizer = MainVisualizer(batched_reference, model, sampler)
    print(sampler.acceptance_ratio())
    visualizer.run()
