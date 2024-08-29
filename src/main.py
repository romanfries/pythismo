from pathlib import Path

import os

import numpy as np

import custom_io
from model.PointDistribution import PointDistributionModel
from src.sampling.Metropolis import PDMMetropolisSampler
from src.custom_io.H5ModelIO import ModelReader
from src.mesh.TMesh import BatchTorchMesh, create_artificial_partial_target
from src.sampling.proposals.ClosestPoint import ClosestPointProposal
from src.sampling.proposals.GaussRandWalk import GaussianRandomWalkProposal, ParameterProposalType
from visualization.DashViewer import MainVisualizer


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
        target.set_points(model.get_points_from_parameters(1.0 * np.ones(model.rank)))
        target = create_artificial_partial_target(target)
        reference.set_points(model.get_points_from_parameters(np.zeros(model.rank)))
        batched_reference = BatchTorchMesh(reference, 'reference', batch_size=2)

        random_walk = GaussianRandomWalkProposal(batched_reference.batch_size, np.zeros(model.rank))
        random_walk_2 = ClosestPointProposal(batched_reference.batch_size, np.zeros(model.rank), reference,
                                             batched_reference, target, model)
        sampler = PDMMetropolisSampler(model, random_walk_2, batched_reference, target, correspondences=False)
        generator = np.random.default_rng()
        for i in range(101):
            random = generator.random()
            if random < 1.0:
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

        model = PointDistributionModel(meshes=meshes)

        if simplify_model:
            reference = model.decimate(1000)
        else:
            reference = meshes[0]

        # TODO: Think again: Does it make sense to represent the target as a TorchMesh?
        target = reference.copy()
        target.set_points(model.get_points_from_parameters(1.0 * np.ones(model.rank)))
        target = create_artificial_partial_target(target)
        reference.set_points(model.get_points_from_parameters(np.zeros(model.rank)))
        batched_reference = BatchTorchMesh(reference, 'reference', batch_size=2)

        random_walk = GaussianRandomWalkProposal(batched_reference.batch_size, np.zeros(model.rank))
        random_walk_2 = ClosestPointProposal(batched_reference.batch_size, np.zeros(model.rank), reference,
                                             batched_reference, target, model)
        sampler = PDMMetropolisSampler(model, random_walk, batched_reference, target, correspondences=False)
        generator = np.random.default_rng()
        for i in range(5):
            random = generator.random()
            if random < 1.0:
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
        pass


if __name__ == "__main__":
    # torch.set_default_tensor_type(torch.cuda.FloatTensor)
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
                                            read_model=False,
                                            simplify_model=True
                                            )
    visualizer = MainVisualizer(batched_reference, model, sampler)
    print(sampler.acceptance_ratio())
    visualizer.run()
