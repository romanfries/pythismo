from pathlib import Path
import os

import torch
from tqdm import tqdm

import custom_io
from model.PointDistribution import PointDistributionModel
from src.registration.IterativeClosestPoints import ICPAnalyser, ICPMode
from src.sampling.Metropolis import PDMMetropolisSampler
from src.custom_io.H5ModelIO import ModelReader
from src.mesh.TMesh import BatchTorchMesh, TorchMeshGpu
from src.sampling.proposals.ClosestPoint import ClosestPointProposal, FullClosestPointProposal
from src.sampling.proposals.GaussRandWalk import GaussianRandomWalkProposal, ParameterProposalType
from visualization.DashViewer import MainVisualizer

# Important notes for running Pytorch3d on Windows:
# https://stackoverflow.com/questions/62304087/installing-pytorch3d-fails-with-anaconda-and-pip-on-windows-10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

read_in = True
simplify = True
path_mesh = "datasets/femur-data/project-data/registered"
path_model = "datasets/femur-data/project-data/models"
path_reference = "datasets/femur-data/project-data/reference-decimated"

BATCH_SIZE = 5
CHAIN_LENGTH = 200001
DECIMATION_TARGET = 1000

MODEL_PROBABILITY = 1.0
TRANSLATION_PROBABILITY = 0.0
ROTATION_PROBABILITY = 0.0

proposal_type = "CP_SIMPLE"


def run(dev, mesh_path=None, reference_path=None, model_path=None, read_model=False, simplify_model=False):
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    if read_model:
        rel_model_path = Path(model_path)
        model_path = Path.cwd().parent / rel_model_path
        model_reader = ModelReader(model_path)
        curr_shape = custom_io.read_meshes(reference_path, dev)[0][0]
        model = model_reader.get_model(dev)
        batched_model = model.get_batched_pdm(BATCH_SIZE)
    else:
        meshes, _ = custom_io.read_meshes(mesh_path, dev)
        model = PointDistributionModel(meshes=meshes)
        if simplify_model:
            curr_shape = model.decimate(DECIMATION_TARGET)
        else:
            curr_shape = meshes[0]
        batched_model = model.get_batched_pdm(BATCH_SIZE)
    # Manually determined landmark points to create a partial target
    target = curr_shape.copy()
    target.set_points(model.get_points_from_parameters(1.0 * torch.ones(model.rank, device=dev)), reset_com=True)
    target = target.partial_shape(4, 195, 0.5)
    batched_target = BatchTorchMesh(target, 'target', dev, BATCH_SIZE)
    params = torch.zeros(model.rank, device=dev)
    batch_params = params.unsqueeze(-1).expand(-1, BATCH_SIZE)
    curr_shape.set_points(model.get_points_from_parameters(params), reset_com=True)
    batched_curr_shape = BatchTorchMesh(curr_shape, 'current_shapes', dev, BATCH_SIZE)
    if proposal_type == "GAUSS_RAND":
        random_walk = GaussianRandomWalkProposal(BATCH_SIZE, params, dev)
        sampler = PDMMetropolisSampler(model, random_walk, batched_curr_shape, batched_target, correspondences=False)
    elif proposal_type == "CP_SIMPLE":
        random_walk = ClosestPointProposal(BATCH_SIZE, params, dev, batched_curr_shape, batched_target,
                                           model)
        sampler = PDMMetropolisSampler(model, random_walk, batched_curr_shape, batched_target, correspondences=False)
    elif proposal_type == "CP_FULL":
        random_walk = FullClosestPointProposal(BATCH_SIZE, batch_params, dev, batched_curr_shape, batched_target,
                                               batched_model)
        sampler = PDMMetropolisSampler(model, random_walk, batched_curr_shape, batched_target, correspondences=False)

    icp = ICPAnalyser(batched_target, batched_curr_shape, mode=ICPMode.BATCHED)
    generator = torch.Generator(device=dev)
    for i in tqdm(range(CHAIN_LENGTH)):
        if CHAIN_LENGTH % 1000 == 0:
            icp.icp()
        random = torch.rand(1, device=dev, generator=generator).item()
        if random < MODEL_PROBABILITY:
            proposal = ParameterProposalType.MODEL
        elif MODEL_PROBABILITY <= random < MODEL_PROBABILITY + TRANSLATION_PROBABILITY:
            proposal = ParameterProposalType.TRANSLATION
        else:
            proposal = ParameterProposalType.ROTATION
        sampler.propose(proposal)
        sampler.determine_quality(proposal)
        sampler.decide(proposal)

    return batched_curr_shape, model, sampler


class Main:
    def __init__(self):
        self.device = device


if __name__ == "__main__":
    # torch.set_default_tensor_type(torch.cuda.FloatTensor)
    main = Main()
    dev = main.device
    batched_reference, model, sampler = run(dev, path_mesh, path_reference, path_model, read_in, simplify)
    batched_reference.change_device(torch.device("cpu"))
    model.change_device(torch.device("cpu"))
    sampler.change_device(torch.device("cpu"))
    visualizer = MainVisualizer(batched_reference, model, sampler)
    acceptance_ratios = sampler.acceptance_ratio()
    print("Acceptance Ratios:")
    strings = ['Parameters', 'Translation', 'Rotation', 'Total']
    for desc, val in zip(strings, acceptance_ratios):
        print(f"{desc}: {val:.4f}")
    visualizer.run()
