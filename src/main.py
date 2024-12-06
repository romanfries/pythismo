import copy
import math
import os
import sys
from operator import attrgetter
from pathlib import Path

import itertools
from tqdm import tqdm

import torch
import torch.multiprocessing as mp

# Absolute imports
current = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(current)
if parent not in sys.path:
    sys.path.insert(0, parent)

from src.analysis import ChainAnalyser
from src.custom_io import DataHandler, read_and_simplify_registered_meshes, read_meshes
from src.mesh import BatchTorchMesh
from src.model import PointDistributionModel, distance_to_closest_point
from src.registration import ICPAnalyser
from src.sampling import PDMMetropolisSampler, create_target_aware_model, create_target_unaware_model
from src.sampling.proposals import ClosestPointProposal
from src.sampling.proposals.GaussRandWalk import ParameterProposalType, GaussianRandomWalkProposal
from src.visualization import MainVisualizer

# TODO: Create a Python package to avoid absolute imports

# Important notes for running Pytorch3d on Windows:
# https://stackoverflow.com/questions/62304087/installing-pytorch3d-fails-with-anaconda-and-pip-on-windows-10

DECIMATE_MESHES = False
RUN_ON_SCICORE_CLUSTER = False
RUN_WHOLE_EXPERIMENT = True
GENERATE_PLOTS = False
# Only relevant, if GENERATE_PLOTS is True. If True, plots are generated that show the average point wise
# distances/variances as a function of an additional parameter, e.g., the variance used for evaluating the likelihood
# term, for every observed percentage. If False, the average point wise distances/variances are plotted against the
# observed percentage.
SEPARATE_PLOTS = False

if RUN_ON_SCICORE_CLUSTER:
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_GPUS = 1
GPU_IDENTIFIERS = list(range(NUM_GPUS))

REL_PATH_MESH = "datasets/femur-data/project-data/registered"
REL_PATH_MESH_DECIMATED = "datasets/femur-data/project-data/registered-decimated"
REL_PATH_INPUT_OUTPUT = "datasets/femur-data/project-data/output/gamma-200-una-std-reg-5"

DISTAL_END = True
ASSUME_FIXED_CORRESPONDENCES = False
MODEL_TARGET_AWARE = False
LAPLACIAN_TYPE = "none"
ALPHA = 1
BETA = 0.8
IDENTITY = 1.0
LANDMARK_TOP, LANDMARK_BOTTOM = 2, 184

BATCH_SIZE = 10
CHAIN_LENGTH = 15000
DEFAULT_BURN_IN = 5000
DECIMATION_TARGET = 200

MODEL_INFORMED_PROBABILITY = 0.5
MODEL_RANDOM_PROBABILITY = 0.1
TRANSLATION_PROBABILITY = 0.2
ROTATION_PROBABILITY = 0.2

VAR_MOD_RANDOM = torch.tensor([0.01, 0.02, 0.04], device=DEVICE)
# VAR_MOD_RANDOM = torch.tensor([0.05, 0.1, 0.2], device=DEVICE)
VAR_MOD_INFORMED = torch.tensor([0.04, 0.08, 0.16], device=DEVICE)
# VAR_MOD_INFORMED = torch.tensor([0.12, 0.24, 0.48], device=DEVICE)
VAR_TRANS = torch.tensor([0.08, 0.16, 0.32], device=DEVICE)
# VAR_TRANS = torch.tensor([0.25, 0.5, 1.0], device=DEVICE)
# Variance in radians
VAR_ROT = torch.tensor([0.0007, 0.0014, 0.0028], device=DEVICE)
# VAR_ROT = torch.tensor([0.002, 0.004, 0.008], device=DEVICE)
PROB_MOD_RANDOM = PROB_MOD_INFORMED = PROB_TRANS = PROB_ROT = torch.tensor([0.2, 0.6, 0.2], device=DEVICE)

UNIFORM_POSE_PRIOR = False
# These two parameters are irrelevant when assuming a uniform pose prior
VAR_PRIOR_TRANS = 3.0
VAR_PRIOR_ROT = 0.005

VAR_LIKELIHOOD_TERM = [1.0]
GAMMA = 200.0

ICP_D = 1.0
ICP_RECALCULATION_PERIOD = 1000

PERCENTAGES_OBSERVED_LENGTH = [0.2]


def memory_stats():
    print(torch.cuda.memory_allocated() / 1024 ** 2)
    print(torch.cuda.memory_reserved() / 1024 ** 2)


def simplify_registered():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    _, _ = read_and_simplify_registered_meshes(REL_PATH_MESH, REL_PATH_MESH_DECIMATED, DEVICE, DECIMATION_TARGET)


def plot():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    handler = DataHandler(REL_PATH_INPUT_OUTPUT)
    # handler.rename_files()
    data_dict = handler.read_all_statistics()
    # Insert list of strings which indicate the chains that have not converged
    chains_to_remove = ['''
    0: 37
    6: 11, 13, 15, 43
    9: 9, 33
    10: 3
    11: 27
    13: 27
    14: 24, 30, 31
    15: 37
    16: 10, 12, 16, 17, 27, 35
    17: 1, 16
    18: 8, 13, 22
    20: 35
    21: 11
    24: 0, 6, 7, 12, 17, 20, 44
    26: 28
    27: 1, 15, 22
    28: 9, 22, 28, 41, 42
    29: 19, 30, 37
    30: 17
    33: 26
    34: 13
    35: 1, 20, 24
    36: 2, 4, 8, 9, 12, 15, 27, 33, 35, 41
    37: 16
    38: 1, 10, 11, 13, 15, 17, 18, 20, 21, 29, 33, 34, 38, 40, 43
    40: 8, 33
    41: 25
    42: 1, 8, 12, 14, 17, 20, 22, 23, 27, 33
    45: 23, 27''', '''
    0: 14
    1: 11
    2: 3
    3: 38
    7: 13, 14, 23, 34, 35
    9: 17, 18
    11: 2, 4, 7, 10
    13: 8, 26, 34
    14: 38
    16: 4, 36
    17: 5
    18: 1, 12, 22, 25
    20: 8
    27: 27, 36
    28: 8, 9, 19, 31
    30: 21
    33: 38
    36: 6, 7, 9, 13, 20, 25, 31, 38, 41
    37: 36
    38: 15, 40
    39: 5
    40: 7
    42: 21''', '''
    6: 7
    9: 36
    10: 29
    11: 6, 28
    13: 8, 20
    14: 28
    16: 13, 19, 20, 22, 27, 29, 35, 37, 38
    19: 1, 34
    20: 33
    21: 31
    24: 19
    25: 8, 17
    27: 24, 25
    29: 3, 13
    30: 3
    34: 2
    35: 2, 15, 30, 32, 33, 34, 35, 36, 41, 43
    38: 15, 22, 27, 30
    42: 22''', '''
    0: 29
    11: 5, 41
    16: 23
    22: 26
    30: 14
    32: 2
    33: 3, 35
    34: 13
    37: 41
    38: 34''', '''
    16: 9
    26: 7''', '''
    ''']
    handler.generate_plots(30, data_dict, chains_to_remove, add_param_available=SEPARATE_PLOTS)


def trial():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # Test procedure with subsequent visualisation
    meshes, _ = read_meshes(REL_PATH_MESH_DECIMATED, DEVICE)
    # loo = torch.randint(0, len(meshes), (1,)).item()
    loo = 0
    # obs = PERCENTAGES_OBSERVED_LENGTH[torch.randint(0, len(PERCENTAGES_OBSERVED_LENGTH), (1,)).item()]
    obs = 0.2
    var_likelihood = 1.0
    distal_end = True

    var_mod_random = math.sqrt(var_likelihood) * VAR_MOD_RANDOM
    var_mod_informed = VAR_MOD_INFORMED
    var_trans = math.sqrt(var_likelihood) * VAR_TRANS
    var_rot = math.sqrt(var_likelihood) * VAR_ROT

    var_prior_trans = VAR_PRIOR_TRANS
    var_prior_rot = VAR_PRIOR_ROT

    target = meshes[loo]
    # z_min, z_max = torch.min(target.tensor_points, dim=0)[1][2].item(), \
    #     torch.max(target.tensor_points, dim=0)[1][2].item()
    # Manually determined landmarks
    z_min, z_max = LANDMARK_BOTTOM, LANDMARK_TOP
    part_target, plane_normal, plane_origin = target.partial_shape(z_max, z_min, obs, distal_end)
    del meshes[loo]
    if MODEL_TARGET_AWARE:
        model, fid, bc = create_target_aware_model(meshes, DEVICE, target, part_target)
    else:
        model, fid, bc = create_target_unaware_model(meshes, DEVICE, part_target)
    shape = meshes[0]
    # dists = distance_to_closest_point(target.tensor_points.unsqueeze(-1), part_target.tensor_points, 1)
    # observed = (dists < 1e-6).squeeze()
    batched_target = BatchTorchMesh(part_target, 'target', DEVICE, BATCH_SIZE)
    starting_params = torch.randn((model.rank, BATCH_SIZE), device=DEVICE)
    starting_translation = VAR_PRIOR_TRANS * torch.randn((3, BATCH_SIZE), device=DEVICE)
    starting_rotation = VAR_PRIOR_ROT * torch.randn((3, BATCH_SIZE), device=DEVICE)
    shape.set_points(model.get_points_from_parameters(torch.zeros(model.rank, device=DEVICE)), adjust_rotation_centre=True)
    batched_shape = BatchTorchMesh(shape, 'current_shapes', DEVICE, BATCH_SIZE, True,
                                   model.get_points_from_parameters(starting_params))
    batched_shape.set_rotation_centre(shape.rotation_centre.unsqueeze(1).expand(3, BATCH_SIZE))
    batched_shape.apply_rotation(starting_rotation)
    batched_shape.apply_translation(starting_translation)
    starting_params = torch.vstack((starting_params, starting_translation, starting_rotation))
    # batched_shape = BatchTorchMesh(shape, 'current_shapes', DEVICE, BATCH_SIZE)

    proposal = ClosestPointProposal(BATCH_SIZE, starting_params, DEVICE, batched_shape, batched_target, model,
                                    var_mod_random, var_mod_informed, var_trans, var_rot, PROB_MOD_RANDOM,
                                    PROB_MOD_INFORMED, PROB_TRANS, PROB_ROT, var_n=var_likelihood, d=ICP_D,
                                    recalculation_period=ICP_RECALCULATION_PERIOD)
    if ASSUME_FIXED_CORRESPONDENCES:
        sampler = PDMMetropolisSampler(model, proposal, batched_shape, batched_target, LAPLACIAN_TYPE, ALPHA, BETA, IDENTITY,
                                       fixed_correspondences=True, triangles=fid, barycentric_coords=bc,
                                       gamma=GAMMA, var_like=var_likelihood, uniform_pose_prior=UNIFORM_POSE_PRIOR,
                                       var_prior_trans=var_prior_trans, var_prior_rot=var_prior_rot)
    else:
        sampler = PDMMetropolisSampler(model, proposal, batched_shape, batched_target, LAPLACIAN_TYPE, ALPHA, BETA, IDENTITY,
                                       fixed_correspondences=False, gamma=GAMMA, var_like=var_likelihood,
                                       uniform_pose_prior=UNIFORM_POSE_PRIOR, var_prior_trans=var_prior_trans,
                                       var_prior_rot=var_prior_rot)

    generator = torch.Generator(device=DEVICE)
    for _ in tqdm(range(CHAIN_LENGTH)):
        random = torch.rand(1, device=DEVICE, generator=generator).item()
        if random < MODEL_INFORMED_PROBABILITY:
            proposal_type = ParameterProposalType.MODEL_INFORMED
        elif random - MODEL_INFORMED_PROBABILITY < MODEL_RANDOM_PROBABILITY:
            proposal_type = ParameterProposalType.MODEL_RANDOM
        elif random - MODEL_INFORMED_PROBABILITY - MODEL_RANDOM_PROBABILITY < TRANSLATION_PROBABILITY:
            proposal_type = ParameterProposalType.TRANSLATION
        else:
            proposal_type = ParameterProposalType.ROTATION
        sampler.propose(proposal_type)
        sampler.determine_quality(proposal_type)
        sampler.decide(proposal_type, target)
    proposal.close()
    # batched_full_target = BatchTorchMesh(target, 'target', torch.device("cpu"), BATCH_SIZE)
    # batched_shape.set_points(torch.stack(list(map(attrgetter('tensor_points'), meshes_))).permute(1, 2, 0))
    batched_shape.change_device(torch.device("cpu"))
    model.change_device(torch.device("cpu"))
    sampler.change_device(torch.device("cpu"))
    visualizer = MainVisualizer(batched_shape, model, sampler)
    # visualizer = MainVisualizer(batched_full_target, model, sampler)
    acceptance_ratios = sampler.acceptance_ratio(torch.ones(BATCH_SIZE, dtype=torch.bool, device=DEVICE))
    print("Acceptance Ratios:")
    strings = ['Parameters', 'Random Noise', 'Translation', 'Rotation', 'Total']
    for desc, val in zip(strings, acceptance_ratios):
        print(f"{desc}: {val:.4f}")
    visualizer.run()


def loocv():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # LOOCV (Leave-One-Out Cross-Validation) procedure
    handler = DataHandler(REL_PATH_INPUT_OUTPUT)
    meshes, _ = read_meshes(REL_PATH_MESH_DECIMATED, DEVICE)
    for var_likelihood in VAR_LIKELIHOOD_TERM:
        var_mod_random = math.sqrt(var_likelihood) * VAR_MOD_RANDOM
        var_mod_informed = VAR_MOD_INFORMED
        var_trans = math.sqrt(var_likelihood) * VAR_TRANS
        var_rot = math.sqrt(var_likelihood) * VAR_ROT

        var_prior_trans = VAR_PRIOR_TRANS
        var_prior_rot = VAR_PRIOR_ROT
        for percentage in PERCENTAGES_OBSERVED_LENGTH:
            for l_ in range(len(meshes)):
            # for l_ in range(1):
                # for l_ in range(10):
                meshes_ = copy.deepcopy(meshes)
                target = meshes_[l_]
                z_min, z_max = LANDMARK_BOTTOM, LANDMARK_TOP
                part_target, plane_normal, plane_origin = target.partial_shape(z_max, z_min, percentage, DISTAL_END)
                del meshes_[l_]
                if MODEL_TARGET_AWARE:
                    model, fid, bc = create_target_aware_model(meshes_, DEVICE, target, part_target)
                else:
                    model, fid, bc = create_target_unaware_model(meshes_, DEVICE, part_target)
                shape = meshes_[0]
                dists = distance_to_closest_point(target.tensor_points.unsqueeze(-1), part_target.tensor_points, 1)
                observed = (dists < 1e-6).squeeze()
                batched_target = BatchTorchMesh(part_target, 'target', DEVICE, BATCH_SIZE)
                starting_params = torch.randn((model.rank, BATCH_SIZE), device=DEVICE)
                starting_translation = VAR_PRIOR_TRANS * torch.zeros((3, BATCH_SIZE), device=DEVICE)
                starting_rotation = VAR_PRIOR_ROT * torch.zeros((3, BATCH_SIZE), device=DEVICE)
                shape.set_points(model.get_points_from_parameters(torch.zeros(model.rank, device=DEVICE)), adjust_rotation_centre=True)
                batched_shape = BatchTorchMesh(shape, 'current_shapes', DEVICE, BATCH_SIZE, True,
                                               model.get_points_from_parameters(starting_params))
                batched_shape.set_rotation_centre(shape.rotation_centre.unsqueeze(1).expand(3, BATCH_SIZE))
                batched_shape.apply_rotation(starting_rotation)
                batched_shape.apply_translation(starting_translation)
                starting_params = torch.vstack((starting_params, starting_translation, starting_rotation))
                # batched_shape = BatchTorchMesh(shape, 'current_shapes', DEVICE, BATCH_SIZE)

                proposal = ClosestPointProposal(BATCH_SIZE, starting_params, DEVICE, batched_shape, batched_target,
                                                model, var_mod_random, var_mod_informed, var_trans, var_rot,
                                                PROB_MOD_RANDOM, PROB_MOD_INFORMED, PROB_TRANS, PROB_ROT,
                                                var_n=var_likelihood, d=ICP_D,
                                                recalculation_period=ICP_RECALCULATION_PERIOD)

                if ASSUME_FIXED_CORRESPONDENCES:
                    sampler = PDMMetropolisSampler(model, proposal, batched_shape, batched_target, LAPLACIAN_TYPE,
                                                   ALPHA, BETA, IDENTITY,
                                                   fixed_correspondences=True, triangles=fid, barycentric_coords=bc,
                                                   gamma=GAMMA, var_like=var_likelihood,
                                                   uniform_pose_prior=UNIFORM_POSE_PRIOR,
                                                   var_prior_trans=var_prior_trans, var_prior_rot=var_prior_rot, save_full_mesh_chain=True,
                                                   save_residuals=True)
                else:
                    sampler = PDMMetropolisSampler(model, proposal, batched_shape, batched_target, LAPLACIAN_TYPE,
                                                   ALPHA, BETA, IDENTITY,
                                                   fixed_correspondences=False, gamma=GAMMA, var_like=var_likelihood,
                                                   uniform_pose_prior=UNIFORM_POSE_PRIOR,
                                                   var_prior_trans=var_prior_trans,
                                                   var_prior_rot=var_prior_rot, save_full_mesh_chain=True, save_residuals=True)

                generator = torch.Generator(device=DEVICE)
                for _ in tqdm(range(CHAIN_LENGTH)):
                    random = torch.rand(1, device=DEVICE, generator=generator).item()
                    if random < MODEL_INFORMED_PROBABILITY:
                        proposal_type = ParameterProposalType.MODEL_INFORMED
                    elif random - MODEL_INFORMED_PROBABILITY < MODEL_RANDOM_PROBABILITY:
                        proposal_type = ParameterProposalType.MODEL_RANDOM
                    elif random - MODEL_INFORMED_PROBABILITY - MODEL_RANDOM_PROBABILITY < TRANSLATION_PROBABILITY:
                        proposal_type = ParameterProposalType.TRANSLATION
                    else:
                        proposal_type = ParameterProposalType.ROTATION
                    sampler.propose(proposal_type)
                    sampler.determine_quality(proposal_type)
                    sampler.decide(proposal_type, target)
                proposal.close()

                analyser = ChainAnalyser(sampler, proposal, model, target, observed, sampler.full_chain,
                                         default_burn_in=DEFAULT_BURN_IN)
                # analyser.detect_burn_in()
                data = analyser.data_to_json(10, l_, int(100 * percentage), int(100 * var_likelihood))
                traceplots = analyser.get_traceplots(l_, int(100 * percentage), int(100 * var_likelihood))
                meshes_.insert(l_, target)
                handler.write_statistics(data, l_, int(100 * percentage), int(100 * var_likelihood))
                handler.write_traceplots(traceplots, l_, int(100 * percentage), int(100 * var_likelihood))
                handler.save_posterior_samples(analyser.mesh_chain[:, :, :, analyser.burn_in:], batched_shape,
                                               part_target, 20, l_, int(100 * percentage), int(100 * var_likelihood),
                                               save_html=True)
                mean_dist_c_map = torch.tensor(data['accuracy']['mean_dist_corr_map'])
                avg_var = torch.tensor(data['accuracy']['var']).mean(dim=1)
                handler.save_target_map_dist(target, mean_dist_c_map, l_, int(100 * percentage),
                                             int(100 * var_likelihood), save_html=True)
                handler.save_target_avg(target, avg_var, l_, int(100 * percentage),
                                             int(100 * var_likelihood), save_html=True)
                # Delete everything that is potentially located on the GPU
                del analyser, avg_var, batched_shape, batched_target, bc, dists, fid, mean_dist_c_map, model, observed, part_target, plane_normal, plane_origin, proposal, sampler, shape, starting_params, starting_rotation, starting_translation, target
                torch.cuda.empty_cache()
                # Code to save further data
                # chain_residual_dict = sampler.get_dict_chain_and_residuals()
                # handler.write_chain_and_residuals(chain_residual_dict, l, int(100 * percentage))
                # param_chain_posterior_dict = proposal.get_dict_param_chain_posterior()
                # handler.write_param_chain_posterior(param_chain_posterior_dict, l_, int(100 * percentage))


def mcmc_task(gpu_id_, chunk_):
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    torch.cuda.set_device(gpu_id_)
    device_ = torch.device(f'cuda:{gpu_id_}')

    var_mod_random = VAR_MOD_RANDOM.to(device_)
    var_mod_informed = VAR_MOD_INFORMED.to(device_)
    var_trans = VAR_TRANS.to(device_)
    var_rot = VAR_ROT.to(device_)
    prob_mod_random = PROB_MOD_RANDOM.to(device_)
    prob_mod_informed = PROB_MOD_INFORMED.to(device_)
    prob_trans = PROB_TRANS.to(device_)
    prob_rot = PROB_ROT.to(device_)

    # LOOCV (Leave-One-Out Cross-Validation) procedure
    handler = DataHandler(REL_PATH_INPUT_OUTPUT)
    meshes, _ = read_meshes(REL_PATH_MESH_DECIMATED, device_)
    for var_likelihood, percentage, l_ in chunk_:
        var_m_rd = math.sqrt(var_likelihood) * var_mod_random
        var_m_in = var_mod_informed
        var_t = math.sqrt(var_likelihood) * var_trans
        var_r = math.sqrt(var_likelihood) * var_rot

        var_prior_trans = VAR_PRIOR_TRANS
        var_prior_rot = VAR_PRIOR_ROT
        # for l_ in range(len(meshes)):
        # for l_ in range(41, 47):
        meshes_ = copy.deepcopy(meshes)
        target = meshes_[l_]
        # z_min, z_max = torch.min(target.tensor_points, dim=0)[1][2].item(), \
        #    torch.max(target.tensor_points, dim=0)[1][2].item()
        z_min, z_max = LANDMARK_BOTTOM, LANDMARK_TOP
        part_target, plane_normal, plane_origin = target.partial_shape(z_max, z_min, percentage, DISTAL_END)
        del meshes_[l_]
        if MODEL_TARGET_AWARE:
            model, fid, bc = create_target_aware_model(meshes_, device_, target, part_target)
        else:
            model, fid, bc = create_target_unaware_model(meshes_, device_, part_target)
        shape = meshes_[0]
        dists = distance_to_closest_point(target.tensor_points.unsqueeze(-1), part_target.tensor_points, 1)
        observed = (dists < 1e-6).squeeze()
        batched_target = BatchTorchMesh(part_target, 'target', device_, BATCH_SIZE)
        starting_params = torch.randn((model.rank, BATCH_SIZE), device=device_)
        starting_translation = VAR_PRIOR_TRANS * torch.randn((3, BATCH_SIZE), device=device_)
        starting_rotation = VAR_PRIOR_ROT * torch.randn((3, BATCH_SIZE), device=device_)
        shape.set_points(model.get_points_from_parameters(torch.zeros(model.rank, device=device_)), adjust_rotation_centre=True)
        batched_shape = BatchTorchMesh(shape, 'current_shapes', device_, BATCH_SIZE, True,
                                       model.get_points_from_parameters(starting_params))
        batched_shape.set_rotation_centre(shape.rotation_centre.unsqueeze(1).expand(3, BATCH_SIZE))
        batched_shape.apply_rotation(starting_rotation)
        batched_shape.apply_translation(starting_translation)
        starting_params = torch.vstack((starting_params, starting_translation, starting_rotation))
        # batched_shape = BatchTorchMesh(shape, 'current_shapes', DEVICE, BATCH_SIZE)

        proposal = ClosestPointProposal(BATCH_SIZE, starting_params, device_, batched_shape, batched_target, model,
                                        var_m_rd, var_m_in, var_t, var_r, prob_mod_random, prob_mod_informed,
                                        prob_trans, prob_rot, var_n=var_likelihood, d=ICP_D,
                                        recalculation_period=ICP_RECALCULATION_PERIOD)

        if ASSUME_FIXED_CORRESPONDENCES:
            sampler = PDMMetropolisSampler(model, proposal, batched_shape, batched_target, LAPLACIAN_TYPE, ALPHA, BETA, IDENTITY,
                                           fixed_correspondences=True, triangles=fid, barycentric_coords=bc,
                                           gamma=GAMMA, var_like=var_likelihood, uniform_pose_prior=UNIFORM_POSE_PRIOR,
                                           var_prior_trans=var_prior_trans, var_prior_rot=var_prior_rot, save_full_mesh_chain=True,
                                           save_residuals=True)
        else:
            sampler = PDMMetropolisSampler(model, proposal, batched_shape, batched_target, LAPLACIAN_TYPE, ALPHA, BETA, IDENTITY,
                                           fixed_correspondences=False, gamma=GAMMA, var_like=var_likelihood,
                                           uniform_pose_prior=UNIFORM_POSE_PRIOR, var_prior_trans=var_prior_trans,
                                           var_prior_rot=var_prior_rot, save_full_mesh_chain=True, save_residuals=True)

        generator = torch.Generator(device=device_)
        for _ in range(CHAIN_LENGTH):
            random = torch.rand(1, device=device_, generator=generator).item()
            if random < MODEL_INFORMED_PROBABILITY:
                proposal_type = ParameterProposalType.MODEL_INFORMED
            elif random - MODEL_INFORMED_PROBABILITY < MODEL_RANDOM_PROBABILITY:
                proposal_type = ParameterProposalType.MODEL_RANDOM
            elif random - MODEL_INFORMED_PROBABILITY - MODEL_RANDOM_PROBABILITY < TRANSLATION_PROBABILITY:
                proposal_type = ParameterProposalType.TRANSLATION
            else:
                proposal_type = ParameterProposalType.ROTATION
            sampler.propose(proposal_type)
            sampler.determine_quality(proposal_type)
            sampler.decide(proposal_type, target)
        proposal.close()

        analyser = ChainAnalyser(sampler, proposal, model, target, observed, sampler.full_chain,
                                 default_burn_in=DEFAULT_BURN_IN)
        data = analyser.data_to_json(10, l_, 20, int(100 * var_likelihood))
        traceplots = analyser.get_traceplots(l_, 20, int(100 * var_likelihood))
        meshes_.insert(l_, target)
        handler.write_statistics(data, l_, 20, int(100 * var_likelihood))
        handler.write_traceplots(traceplots, l_, 20, int(100 * var_likelihood))
        handler.save_posterior_samples(analyser.mesh_chain[:, :, :, analyser.burn_in:], batched_shape,
                                       part_target, 20, l_, 20, int(100 * var_likelihood))
        mean_dist_c_map = torch.tensor(data['accuracy']['mean_dist_corr_map'])
        avg_var = torch.tensor(data['accuracy']['var']).mean(dim=1)
        handler.save_target_map_dist(target, mean_dist_c_map, l_, 20, int(100 * var_likelihood),
                                     save_html=True)
        handler.save_target_avg(target, avg_var, l_, 20, int(100 * var_likelihood), save_html=True)

        # Delete everything that is potentially located on the GPU
        del analyser, avg_var, batched_shape, batched_target, bc, dists, fid, mean_dist_c_map, model, observed, part_target, plane_normal, plane_origin, proposal, sampler, shape, starting_params, starting_rotation, starting_translation, target
        torch.cuda.empty_cache()
        # Code to save further data
        # chain_residual_dict = sampler.get_dict_chain_and_residuals()
        # handler.write_chain_and_residuals(chain_residual_dict, l, int(100 * percentage))
        # param_chain_posterior_dict = random_walk.get_dict_param_chain_posterior()
        # handler.write_param_chain_posterior(param_chain_posterior_dict, l, int(100 * percentage))


if __name__ == "__main__":
    if DECIMATE_MESHES:
        simplify_registered()
    if RUN_WHOLE_EXPERIMENT:
        if RUN_ON_SCICORE_CLUSTER:
            mp.set_start_method('spawn')
            p = (Path.cwd().parent / Path(REL_PATH_MESH_DECIMATED)).glob('**/*')
            mesh_list = [f for f in p if f.is_file()]
            tasks = list(itertools.product(VAR_LIKELIHOOD_TERM, PERCENTAGES_OBSERVED_LENGTH, range(len(mesh_list))))
            # tasks = [(1.0, 0.2, 4)]
            chunks = [tasks[i::NUM_GPUS] for i in range(NUM_GPUS)]
            processes = []
            for gpu_id, chunk in zip(GPU_IDENTIFIERS, chunks):
                process = mp.Process(target=mcmc_task, args=(gpu_id, chunk))
                process.start()
                processes.append(process)

            for p in processes:
                p.join()
        else:
            loocv()
    elif GENERATE_PLOTS:
        plot()
    else:
        trial()
