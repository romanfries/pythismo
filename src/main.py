import threading
import time
import warnings
from pathlib import Path

import os
import numpy as np
from flask import request

import custom_io
from custom_io.MeshIO import MeshReaderWriter
from model.PointDistribution import PointDistributionModel
from src.sampling.Metropolis import PDMMetropolisSampler
from registration.Procrustes import ProcrustesAnalyser
from src.custom_io.H5ModelIO import ModelReader
from src.mesh.TMesh import BatchTorchMesh
from src.registration.IterativeClosestPoints import ICPAnalyser
from src.sampling.proposals.GaussRandWalk import GaussianRandomWalkProposal, ParameterProposalType
from visualization.DashViewer import MainVisualizer


def run(mesh_path,
        landmark_path=None,
        reference_lm_path=None,
        reference_path=None,
        model_path=None,
        read_model=False,
        simplify_model=False,
        registered=False,
        write_meshes=False):
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    if read_model:
        rel_model_path = Path(model_path)
        model_path = Path.cwd().parent / rel_model_path
        model_reader = ModelReader(model_path)
        reference = custom_io.read_meshes(reference_path)[0]
        model = model_reader.get_model()
        target = reference.copy()
        target.set_points(model.get_points_from_parameters(3.0*np.ones(model.sample_size)))
        reference.set_points(model.get_points_from_parameters(np.zeros(model.sample_size)))
        batched_reference = BatchTorchMesh(reference, 'reference', batch_size=2)

        random_walk = GaussianRandomWalkProposal(batched_reference.batch_size, np.zeros(model.sample_size))
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

        if not registered:
            ref, landmarks = custom_io.read_landmarks_with_reference(reference_lm_path, landmark_path)

            landmark_aligner = ProcrustesAnalyser(landmarks, ref)
            transforms, _, identifiers = landmark_aligner.generalised_procrustes_alignment()

            dict_transforms = {identifier: transform for transform, identifier in zip(transforms, identifiers)}
            for i, mesh in enumerate(meshes):
                if mesh.id in dict_transforms:
                    mesh.apply_transformation(dict_transforms.get(mesh.id))
                    meshes[i] = mesh.simplify_qem(target=200)
                else:
                    warnings.warn("Warning: Unexpected error occurred while transforming the meshes.",
                                  UserWarning)

            meshes = [mesh for mesh in meshes if mesh.id != 'reference']
            icp_aligner = ICPAnalyser(meshes)
            icp_aligner.icp()

        if write_meshes:
            write_path = Path().cwd().parent / Path(mesh_path).parent / 'meshes-prepared-2'
            write_path.mkdir(parents=True, exist_ok=True)
            for mesh in meshes:
                file_name = str(mesh.id) + '.stl'
                mesh_io = MeshReaderWriter(write_path / file_name)
                mesh_io.write_mesh(mesh)

        model = PointDistributionModel(meshes)

        if simplify_model:
            reference = model.decimate(200)
        else:
            reference = meshes[0]

        target = reference.copy()
        target.set_points(model.get_points_from_parameters(3.0 * np.ones(model.sample_size)))
        reference.set_points(model.get_points_from_parameters(np.zeros(model.sample_size)))
        batched_reference = BatchTorchMesh(reference, 'reference', batch_size=2)

        random_walk = GaussianRandomWalkProposal(batched_reference.batch_size, np.zeros(model.sample_size))
        sampler = PDMMetropolisSampler(model, random_walk, batched_reference, target, correspondences=True)
        generator = np.random.default_rng()
        for i in range(1001):
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


def dash(visualizer):
    visualizer.run()


class Main:
    def __init__(self):
        pass


if __name__ == "__main__":
    main = Main()
    batched_reference, model, sampler = run("datasets/femur-data/project-data/registered",
        landmark_path="datasets/femur-data/project-data/landmarks",
        reference_lm_path="datasets/femur-data/project-data/reference-landmarks",
        reference_path="datasets/femur-data/project-data/reference-decimated",
        model_path="datasets/models",
        read_model=False,
        simplify_model=True,
        registered=True,
        write_meshes=False
        )
    visualizer = MainVisualizer(batched_reference, model, sampler)
    # print(converter.acceptance_ratio())
    dash_thread = threading.Thread(target=dash(visualizer))
    dash_thread.start()
    dash_thread.run()

    try:
        while True:
            print("Dash server running in the background.")
            time.sleep(100)
    except KeyboardInterrupt:
        import requests

        requests.post('http://127.0.0.1:8050/shutdown')
        dash_thread.join()
