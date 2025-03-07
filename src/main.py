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
RUN_WHOLE_EXPERIMENT = False
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
REL_PATH_MESH_DECIMATED = "datasets/femur-data/project-data/registered-decimated-2000"
REL_PATH_INPUT_OUTPUT = "datasets/femur-data/project-data/output/long-high-0"

DISTAL_END = True
ASSUME_FIXED_CORRESPONDENCES = False
MODEL_TARGET_AWARE = False
LAPLACIAN_TYPE = "std"
ALPHA = 1
BETA = 200
IDENTITY = 1.0
LANDMARK_TOP, LANDMARK_BOTTOM = 16, 1752

BATCH_SIZE = 45
CHAIN_LENGTH = 50000
DEFAULT_BURN_IN = 1000
DECIMATION_TARGET = 2000

MODEL_INFORMED_PROBABILITY = 0.5
MODEL_RANDOM_PROBABILITY = 0.1
TRANSLATION_PROBABILITY = 0.2
ROTATION_PROBABILITY = 0.2

VAR_MOD_RANDOM = torch.tensor([0.02, 0.04, 0.08], device=DEVICE)
VAR_MOD_INFORMED = torch.tensor([0.08, 0.16, 0.32], device=DEVICE)
VAR_TRANS = torch.tensor([0.1, 0.2, 0.4], device=DEVICE)
# Variance in radians
VAR_ROT = torch.tensor([0.001, 0.002, 0.004], device=DEVICE)
PROB_MOD_RANDOM = PROB_MOD_INFORMED = PROB_TRANS = PROB_ROT = torch.tensor([0.2, 0.6, 0.2], device=DEVICE)

UNIFORM_POSE_PRIOR = False
# These two parameters are irrelevant when assuming a uniform pose prior
VAR_PRIOR_TRANS = 20.0
VAR_PRIOR_ROT = 0.005

VAR_LIKELIHOOD_TERM = [1.0]
GAMMA = 100.0

ICP_D = 1.0
ICP_RECALCULATION_PERIOD = 50000

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
            3: 17, 20
            5: 5
            6: 4
            8: 15
            9: 11
            18: 8
            22: 12
            25: 10
            28: 19
            42: 9
            45: 18, 19''', '''
        	0: 15
            7: 5, 13
            12: 8
            15: 6
            17: 6
            20: 7
            22: 7, 14
            39: 10
            41: 6
            43: 14
            45: 0''', '''3: 17
            4: 11
            6: 13
            11: 1, 3
            17: 14
            21: 4
            28: 5, 10
            29: 15
            46: 11''', '''2: 6
            6: 2
            7: 26
            11: 19
            22: 8
            29: 1
            37: 23
            42: 18
            43: 6
            46: 24''', '''4: 2
            5: 20
            12: 28
            13: 14
            14: 16
            15: 30
            23: 0
            24: 29
            28: 20
            29: 20
            37: 18
            39: 32
            42: 25''', '''0: 4
            5: 12, 27
            9: 29
            10: 27
            11: 14
            16: 10
            17: 11
            20: 10
            24: 12
            27: 1
            29: 7, 25
            37: 6, 21
            38: 1''']
    handler.generate_plots(11, data_dict, chains_to_remove, add_param_available=SEPARATE_PLOTS)


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
    # meshes = [meshes[i] for i in [21, 41, 45]]
    # disturb = torch.tensor([50.0, 0.0, 0.0], device=DEVICE)
    # meshes[1].set_points(meshes[1].tensor_points - disturb)
    # meshes[2].set_points(meshes[2].tensor_points + disturb)
    points = torch.stack(list(map(attrgetter('tensor_points'), meshes)))
    # meshes[1].set_points(meshes[1].tensor_points + disturb)
    # meshes[2].set_points(meshes[2].tensor_points - disturb)
    if MODEL_TARGET_AWARE:
        model, fid, bc, pts = create_target_aware_model(meshes, DEVICE, target, part_target)
    else:
        model, fid, bc, pts = create_target_unaware_model(meshes, DEVICE, part_target)
    # points = torch.vstack((points, pts))
    shape = meshes[0]
    dists = distance_to_closest_point(target.tensor_points.unsqueeze(-1), part_target.tensor_points, 1)
    observed = (dists < 1e-6).squeeze()
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
        sampler = PDMMetropolisSampler(model, proposal, batched_shape, batched_target, LAPLACIAN_TYPE,
                                       ALPHA, BETA, IDENTITY,
                                       fixed_correspondences=False, gamma=GAMMA, var_like=var_likelihood,
                                       uniform_pose_prior=UNIFORM_POSE_PRIOR,
                                       var_prior_trans=var_prior_trans,
                                       var_prior_rot=var_prior_rot, save_full_mesh_chain=False, save_residuals=False,
                                       metrics_on_the_fly=True, full_target=target, default_burn_in=DEFAULT_BURN_IN)

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
    sampler.close()
    # batched_full_target = BatchTorchMesh(target, 'target', torch.device("cpu"), BATCH_SIZE)
    # batched_full_target.set_points(points.permute(1, 2, 0))
    # batched_full_target.change_device("cpu")
    # batched_shape.set_points(torch.stack(list(map(attrgetter('tensor_points'), meshes_))).permute(1, 2, 0))
    # batched_shape.change_device(torch.device("cpu"))
    model.change_device(torch.device("cpu"))
    sampler.change_device(torch.device("cpu"))
    visualizer = MainVisualizer(batched_shape, model, sampler, observed.cpu())
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
                    model, fid, bc, pts = create_target_aware_model(meshes_, DEVICE, target, part_target)
                else:
                    model, fid, bc, pts = create_target_unaware_model(meshes_, DEVICE, part_target)
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
                                                   var_prior_rot=var_prior_rot, save_full_mesh_chain=False, save_residuals=False,
                                                   metrics_on_the_fly=True, full_target=target, default_burn_in=DEFAULT_BURN_IN)

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
                sampler.close()

                analyser = ChainAnalyser(sampler, proposal, model, target, observed, default_burn_in=DEFAULT_BURN_IN,
                                         recalculate_mesh_chain=False)
                # analyser.detect_burn_in()
                data = analyser.data_to_json(10, l_, int(100 * percentage), int(100 * var_likelihood))
                traceplots = analyser.get_traceplots(l_, int(100 * percentage), int(100 * var_likelihood))
                meshes_.insert(l_, target)
                handler.write_statistics(data, l_, int(100 * percentage), int(100 * var_likelihood))
                handler.write_traceplots(traceplots, l_, int(100 * percentage), int(100 * var_likelihood))
                # handler.save_posterior_samples(analyser.mesh_chain[:, :, :, analyser.burn_in:], batched_shape,
                #                                part_target, 3, l_, int(100 * percentage), int(100 * var_likelihood),
                #                                save_html=True)
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
            model, fid, bc, pts = create_target_aware_model(meshes_, device_, target, part_target)
        else:
            model, fid, bc, pts = create_target_unaware_model(meshes_, device_, part_target)
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
                                           save_residuals=True, variances_on_the_fly=False, default_burn_in=DEFAULT_BURN_IN)
        else:
            sampler = PDMMetropolisSampler(model, proposal, batched_shape, batched_target, LAPLACIAN_TYPE, ALPHA, BETA, IDENTITY,
                                           fixed_correspondences=False, gamma=GAMMA, var_like=var_likelihood,
                                           uniform_pose_prior=UNIFORM_POSE_PRIOR, var_prior_trans=var_prior_trans,
                                           var_prior_rot=var_prior_rot, save_full_mesh_chain=False, save_residuals=False,
                                           metrics_on_the_fly=True, full_target=target, default_burn_in=DEFAULT_BURN_IN)

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
        sampler.close()

        analyser = ChainAnalyser(sampler, proposal, model, target, observed, # sampler.full_chain,
                                 default_burn_in=DEFAULT_BURN_IN, recalculate_mesh_chain=False)
        data = analyser.data_to_json(10, l_, 28, int(100 * var_likelihood))
        traceplots = analyser.get_traceplots(l_, 28, int(100 * var_likelihood))
        meshes_.insert(l_, target)
        handler.write_statistics(data, l_, 28, int(100 * var_likelihood))
        handler.write_traceplots(traceplots, l_, 28, int(100 * var_likelihood))
        # Not implemented for the case that the mesh chain is neither saved nor recalculated
        # handler.save_posterior_samples(analyser.mesh_chain[:, :, :, analyser.burn_in:], batched_shape,
        #                                part_target, 20, l_, 20, int(100 * var_likelihood))
        mean_dist_c_map = torch.tensor(data['accuracy']['mean_dist_corr_map'])
        avg_var = torch.tensor(data['accuracy']['var']).mean(dim=1)
        handler.save_target_map_dist(target, mean_dist_c_map, l_, 28, int(100 * var_likelihood),
                                     save_html=True)
        handler.save_target_avg(target, avg_var, l_, 28, int(100 * var_likelihood), save_html=True)

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
            # tasks = [(1.0, 0.2, 0)]
            # tasks = [(0.4, 0.2, 1), (0.4, 0.2, 5), (0.4, 0.2, 7), (0.4, 0.2, 9), (0.4, 0.2, 11), (0.4, 0.2, 13),
            #          (0.4, 0.2, 14), (0.4, 0.2, 16), (0.4, 0.2, 17), (0.4, 0.2, 19), (0.4, 0.2, 20), (0.4, 0.2, 24),
            #          (0.4, 0.2, 25), (0.4, 0.2, 26), (0.4, 0.2, 27), (0.4, 0.2, 29), (0.4, 0.2, 33), (0.4, 0.2, 35),
            #          (0.4, 0.2, 38), (0.4, 0.2, 39), (0.4, 0.2, 42), (0.4, 0.2, 43), (0.4, 0.2, 46), (0.6, 0.2, 1),
            #          (0.6, 0.2, 7), (0.6, 0.2, 13), (0.6, 0.2, 16), (0.6, 0.2, 27), (0.6, 0.2, 38), (0.8, 0.2, 33),
            #          (2.0, 0.2, 38), (2.0, 0.2, 41)]
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
