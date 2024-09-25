import os
from tqdm import tqdm

import torch

from src.analysis import ChainAnalyser
from src.custom_io import DataHandler, read_and_simplify_registered_meshes, read_meshes
from src.mesh import BatchTorchMesh
from src.model import PointDistributionModel, distance_to_closest_point
from src.registration import ICPAnalyser
from src.sampling import PDMMetropolisSampler
from src.sampling.proposals import ClosestPointProposal
from src.sampling.proposals import GaussianRandomWalkProposal, ParameterProposalType
from src.visualization import MainVisualizer

# Important notes for running Pytorch3d on Windows:
# https://stackoverflow.com/questions/62304087/installing-pytorch3d-fails-with-anaconda-and-pip-on-windows-10

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

RUN_WHOLE_EXPERIMENT = False
GENERATE_PLOTS = False

REL_PATH_MESH = "datasets/femur-data/project-data/registered"
REL_PATH_MESH_DECIMATED = "datasets/femur-data/project-data/registered-decimated"
REL_PATH_MODEL = "datasets/femur-data/project-data/models"
REL_PATH_REFERENCE = "datasets/femur-data/project-data/reference-decimated"
REL_PATH_INPUT_OUTPUT = "datasets/femur-data/project-data/output/cp"

BATCH_SIZE = 50
CHAIN_LENGTH = 12000
DECIMATION_TARGET = 200

MODEL_PROBABILITY = 0.6
TRANSLATION_PROBABILITY = 0.2
ROTATION_PROBABILITY = 0.2

PROPOSAL_TYPE = "CP_SIMPLE"

SIGMA_MOD_GAUSS = torch.tensor([0.04, 0.08, 0.16], device=DEVICE)
SIGMA_MOD_CP = torch.tensor([0.1, 0.2, 0.4], device=DEVICE)
SIGMA_TRANS = torch.tensor([0.15, 0.3, 0.6], device=DEVICE)
# Variance in radians
SIGMA_ROT = torch.tensor([0.001, 0.002, 0.004], device=DEVICE)
PROB_MOD = PROB_TRANS = PROB_ROT = torch.tensor([0.2, 0.6, 0.2], device=DEVICE)

UNIFORM_POSE_PRIOR = False
# These two parameters are irrelevant when selecting a uniform pose prior
SIGMA_PRIOR_TRANS = 30.0
SIGMA_PRIOR_ROT = 0.01

CP_D = 1.0
CP_RECALCULATION_PERIOD = 1000

PERCENTAGES_OBSERVED_LENGTH = [0.2, 0.4, 0.6, 0.8]


def simplify_registered():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    _, _ = read_and_simplify_registered_meshes(REL_PATH_MESH, REL_PATH_MESH_DECIMATED, DEVICE,
                                               DECIMATION_TARGET)


def plot():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    handler = DataHandler(REL_PATH_INPUT_OUTPUT)
    data_dict = handler.read_all_statistics()
    handler.generate_plots(data_dict)


def trial():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # Test procedure with subsequent visualisation
    meshes, _ = read_meshes(REL_PATH_MESH_DECIMATED, DEVICE)
    loo = torch.randint(0, len(meshes), (1,)).item()
    # obs = PERCENTAGES_OBSERVED_LENGTH[torch.randint(0, len(PERCENTAGES_OBSERVED_LENGTH), (1,)).item()]
    obs = 0.2
    target = meshes[loo]
    del meshes[loo]
    z_min, z_max = torch.min(target.tensor_points, dim=0)[1][2].item(), \
        torch.max(target.tensor_points, dim=0)[1][2].item()
    part_target, plane_normal, plane_origin = target.partial_shape(z_max, z_min, obs)
    meshes.insert(0, part_target)
    ICPAnalyser(meshes).icp()
    del meshes[0]
    model = PointDistributionModel(meshes=meshes)
    shape = meshes[0]
    # dists = distance_to_closest_point(target.tensor_points.unsqueeze(-1), part_target.tensor_points, 1)
    # observed = (dists < 1e-6).squeeze()
    batched_target = BatchTorchMesh(part_target, 'target', DEVICE, BATCH_SIZE)
    starting_params = torch.zeros(model.rank, device=DEVICE)
    shape.set_points(model.get_points_from_parameters(starting_params), reset_com=True)
    batched_shape = BatchTorchMesh(shape, 'current_shapes', DEVICE, BATCH_SIZE)
    if PROPOSAL_TYPE == "GAUSS_RAND":
        random_walk = GaussianRandomWalkProposal(BATCH_SIZE, starting_params, DEVICE, SIGMA_MOD_GAUSS,
                                                 SIGMA_TRANS, SIGMA_ROT, PROB_MOD, PROB_TRANS, PROB_ROT)
    elif PROPOSAL_TYPE == "CP_SIMPLE":
        random_walk = ClosestPointProposal(BATCH_SIZE, starting_params, DEVICE, batched_shape,
                                           batched_target, model, SIGMA_MOD_CP, SIGMA_TRANS, SIGMA_ROT, PROB_MOD,
                                           PROB_TRANS, PROB_ROT, CP_D, CP_RECALCULATION_PERIOD)
    sampler = PDMMetropolisSampler(model, random_walk, batched_shape, batched_target, correspondences=False,
                                   uniform_pose_prior=UNIFORM_POSE_PRIOR, sigma_trans=SIGMA_PRIOR_TRANS,
                                   sigma_rot=SIGMA_PRIOR_ROT)

    generator = torch.Generator(device=DEVICE)
    for _ in tqdm(range(CHAIN_LENGTH)):
        random = torch.rand(1, device=DEVICE, generator=generator).item()
        if random < MODEL_PROBABILITY:
            proposal = ParameterProposalType.MODEL
        elif MODEL_PROBABILITY <= random < MODEL_PROBABILITY + TRANSLATION_PROBABILITY:
            proposal = ParameterProposalType.TRANSLATION
        else:
            proposal = ParameterProposalType.ROTATION
        sampler.propose(proposal)
        sampler.determine_quality(proposal)
        sampler.decide(proposal, target)
    random_walk.close()

    batched_shape.change_device(torch.device("cpu"))
    model.change_device(torch.device("cpu"))
    sampler.change_device(torch.device("cpu"))
    visualizer = MainVisualizer(batched_shape, model, sampler)
    acceptance_ratios = sampler.acceptance_ratio()
    print("Acceptance Ratios:")
    strings = ['Parameters', 'Translation', 'Rotation', 'Total']
    for desc, val in zip(strings, acceptance_ratios):
        print(f"{desc}: {val:.4f}")
    visualizer.run()


def trial_legacy():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # Test procedure with subsequent visualisation
    meshes, _ = read_meshes(REL_PATH_MESH, DEVICE)
    loo = torch.randint(0, len(meshes), (1,)).item()
    # obs = PERCENTAGES_OBSERVED_LENGTH[torch.randint(0, len(PERCENTAGES_OBSERVED_LENGTH), (1,)).item()]
    obs = 0.2
    target = meshes[loo]
    del meshes[loo]
    z_min, z_max = torch.min(target.tensor_points, dim=0)[1][2].item(), \
        torch.max(target.tensor_points, dim=0)[1][2].item()
    part_target, plane_normal, plane_origin = target.partial_shape(z_max, z_min, obs)
    meshes.insert(0, part_target)
    ICPAnalyser(meshes).icp()
    del meshes[0]
    model = PointDistributionModel(meshes=meshes)
    shape = model.decimate(DECIMATION_TARGET)
    dec_target = target.simplify_ref(meshes[0], shape)
    dec_part_target, _, _ = dec_target.partial_shape(0, 0, obs, True, plane_normal, plane_origin)
    # dists = distance_to_closest_point(dec_target.tensor_points.unsqueeze(-1), dec_part_target.tensor_points, 1)
    # observed = (dists < 1e-6).squeeze()
    batched_target = BatchTorchMesh(dec_part_target, 'target', DEVICE, BATCH_SIZE)
    starting_params = torch.zeros(model.rank, device=DEVICE)
    shape.set_points(model.get_points_from_parameters(starting_params), reset_com=True)
    batched_shape = BatchTorchMesh(shape, 'current_shapes', DEVICE, BATCH_SIZE)
    if PROPOSAL_TYPE == "GAUSS_RAND":
        random_walk = GaussianRandomWalkProposal(BATCH_SIZE, starting_params, DEVICE, SIGMA_MOD_GAUSS,
                                                 SIGMA_TRANS, SIGMA_ROT, PROB_MOD, PROB_TRANS, PROB_ROT)
    elif PROPOSAL_TYPE == "CP_SIMPLE":
        random_walk = ClosestPointProposal(BATCH_SIZE, starting_params, DEVICE, batched_shape,
                                           batched_target, model, SIGMA_MOD_CP, SIGMA_TRANS, SIGMA_ROT, PROB_MOD,
                                           PROB_TRANS, PROB_ROT, CP_D, CP_RECALCULATION_PERIOD)
    sampler = PDMMetropolisSampler(model, random_walk, batched_shape, batched_target, correspondences=False,
                                   uniform_pose_prior=UNIFORM_POSE_PRIOR, sigma_trans=SIGMA_PRIOR_TRANS,
                                   sigma_rot=SIGMA_PRIOR_ROT)

    generator = torch.Generator(device=DEVICE)
    for _ in tqdm(range(CHAIN_LENGTH)):
        random = torch.rand(1, device=DEVICE, generator=generator).item()
        if random < MODEL_PROBABILITY:
            proposal = ParameterProposalType.MODEL
        elif MODEL_PROBABILITY <= random < MODEL_PROBABILITY + TRANSLATION_PROBABILITY:
            proposal = ParameterProposalType.TRANSLATION
        else:
            proposal = ParameterProposalType.ROTATION
        sampler.propose(proposal)
        sampler.determine_quality(proposal)
        sampler.decide(proposal, dec_target)
    random_walk.close()

    batched_shape.change_device(torch.device("cpu"))
    model.change_device(torch.device("cpu"))
    sampler.change_device(torch.device("cpu"))
    visualizer = MainVisualizer(batched_shape, model, sampler)
    acceptance_ratios = sampler.acceptance_ratio()
    print("Acceptance Ratios:")
    strings = ['Parameters', 'Translation', 'Rotation', 'Total']
    for desc, val in zip(strings, acceptance_ratios):
        print(f"{desc}: {val:.4f}")
    visualizer.run()


def loocv():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # LOOCV (Leave-One-Out Cross-Validation) procedure
    handler = DataHandler(REL_PATH_INPUT_OUTPUT)
    meshes, _ = read_meshes(REL_PATH_MESH, DEVICE)
    for percentage in PERCENTAGES_OBSERVED_LENGTH:
        for l_ in range(len(meshes)):
            target = meshes[l_]
            del meshes[l_]
            z_min, z_max = torch.min(target.tensor_points, dim=0)[1][2].item(), \
                torch.max(target.tensor_points, dim=0)[1][2].item()
            part_target, plane_normal, plane_origin = target.partial_shape(z_max, z_min, percentage)
            meshes.insert(0, part_target)
            ICPAnalyser(meshes).icp()
            del meshes[0]
            model = PointDistributionModel(meshes=meshes)
            shape = meshes[0]
            dists = distance_to_closest_point(target.tensor_points.unsqueeze(-1), part_target.tensor_points, 1)
            observed = (dists < 1e-6).squeeze()
            batched_target = BatchTorchMesh(part_target, 'target', DEVICE, BATCH_SIZE)
            starting_params = torch.zeros(model.rank, device=DEVICE)
            shape.set_points(model.get_points_from_parameters(starting_params), reset_com=True)
            batched_shape = BatchTorchMesh(shape, 'current_shapes', DEVICE, BATCH_SIZE)
            if PROPOSAL_TYPE == "GAUSS_RAND":
                random_walk = GaussianRandomWalkProposal(BATCH_SIZE, starting_params, DEVICE, SIGMA_MOD_GAUSS,
                                                         SIGMA_TRANS, SIGMA_ROT, PROB_MOD, PROB_TRANS, PROB_ROT)
            elif PROPOSAL_TYPE == "CP_SIMPLE":
                random_walk = ClosestPointProposal(BATCH_SIZE, starting_params, DEVICE, batched_shape,
                                                   batched_target, model, SIGMA_MOD_CP, SIGMA_TRANS, SIGMA_ROT,
                                                   PROB_MOD, PROB_TRANS, PROB_ROT, CP_D, CP_RECALCULATION_PERIOD)
            sampler = PDMMetropolisSampler(model, random_walk, batched_shape, batched_target, correspondences=False,
                                           uniform_pose_prior=UNIFORM_POSE_PRIOR, sigma_trans=SIGMA_PRIOR_TRANS,
                                           sigma_rot=SIGMA_PRIOR_ROT, save_full_mesh_chain=True,
                                           save_residuals=True)

            generator = torch.Generator(device=DEVICE)
            for _ in tqdm(range(CHAIN_LENGTH)):
                random = torch.rand(1, device=DEVICE, generator=generator).item()
                if random < MODEL_PROBABILITY:
                    proposal = ParameterProposalType.MODEL
                elif MODEL_PROBABILITY <= random < MODEL_PROBABILITY + TRANSLATION_PROBABILITY:
                    proposal = ParameterProposalType.TRANSLATION
                else:
                    proposal = ParameterProposalType.ROTATION
                sampler.propose(proposal)
                sampler.determine_quality(proposal)
                sampler.decide(proposal, target)
            random_walk.close()

            analyser = ChainAnalyser(sampler, random_walk, model, observed, sampler.full_chain)
            # analyser.detect_burn_in()
            data = analyser.data_to_json(l_, int(100 * percentage))
            meshes.insert(l_, target)
            handler.write_statistics(data, l_, int(100 * percentage))
            # Code to save further data
            # chain_residual_dict = sampler.get_dict_chain_and_residuals()
            # handler.write_chain_and_residuals(chain_residual_dict, l, int(100 * percentage))
            # param_chain_posterior_dict = random_walk.get_dict_param_chain_posterior()
            # handler.write_param_chain_posterior(param_chain_posterior_dict, l, int(100 * percentage))


if __name__ == "__main__":
    simplify_registered()
    if RUN_WHOLE_EXPERIMENT:
        loocv()
    elif GENERATE_PLOTS:
        plot()
    else:
        trial()
