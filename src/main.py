import json
from pathlib import Path
import os

import torch
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# TODO: Organise imports.
import custom_io
from model.PointDistribution import PointDistributionModel, distance_to_closest_point
from src.analysis.Analyser import ChainAnalyser
from src.custom_io.DataIO import DataHandler
from src.registration.IterativeClosestPoints import ICPAnalyser, ICPMode
from src.sampling.Metropolis import PDMMetropolisSampler
from src.custom_io.H5ModelIO import ModelReader
from src.mesh.TMesh import BatchTorchMesh, TorchMeshGpu
from src.sampling.proposals.ClosestPoint import ClosestPointProposal, FullClosestPointProposal
from src.sampling.proposals.GaussRandWalk import GaussianRandomWalkProposal, ParameterProposalType
from visualization.DashViewer import MainVisualizer

# Important notes for running Pytorch3d on Windows:
# https://stackoverflow.com/questions/62304087/installing-pytorch3d-fails-with-anaconda-and-pip-on-windows-10

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

READ_JSON = False

READ_IN = False
SIMPLIFY = True
REL_PATH_MESH = "datasets/femur-data/project-data/registered"
REL_PATH_MODEL = "datasets/femur-data/project-data/models"
REL_PATH_REFERENCE = "datasets/femur-data/project-data/reference-decimated"

REL_INPUT_DIR = "datasets/output/ClosestPointProposal"
REL_OUTPUT_DIR = "datasets/output"

BATCH_SIZE = 20
CHAIN_LENGTH = 22000
DECIMATION_TARGET = 200

MODEL_PROBABILITY = 0.6
TRANSLATION_PROBABILITY = 0.2
ROTATION_PROBABILITY = 0.2

PROPOSAL_TYPE = "GAUSS_RAND"

SIGMA_MOD_GAUSS = 0.05
SIGMA_MOD_CP = 0.2
# Variance (in square millimetres) in the CP is divided by the square root of the number of model points N to calculate
# the actual variance of the perturbations.
SIGMA_TRANS_GAUSS = 0.1
SIGMA_TRANS_CP = 10.0
# Variance in radians
SIGMA_ROT = 0.001

CP_D = 1.0
CP_RECALCULATION_PERIOD = 1000

PERCENTAGES_OBSERVED_LENGTH = [0.2, 0.4, 0.6, 0.8, 1.0]


def run():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    if READ_JSON:
        handler = DataHandler(REL_OUTPUT_DIR, REL_INPUT_DIR)
        data_dict = handler.read_all_jsons()

        avg_var = data_dict['accuracy']['avg_var']
        mean_dist_n = data_dict['accuracy']['mean_dist_n']
        obs = data_dict['identifiers']['obs']
        observed = data_dict['observed']


        avg_var_all = avg_var.mean(dim=1).numpy()
        mean_dist_n_all = mean_dist_n.mean(dim=1).numpy()

        avg_var_copy_t, avg_var_copy_f = avg_var.clone(), avg_var.clone()
        mean_dist_n_copy_t, mean_dist_n_copy_f = mean_dist_n.clone(), mean_dist_n.clone()
        avg_var_copy_t[~observed] = float('nan')
        mean_dist_n_copy_t[~observed] = float('nan')
        avg_var_mean_t = torch.nanmean(avg_var_copy_t, dim=1)
        mean_dist_n_mean_t = torch.nanmean(mean_dist_n_copy_t, dim=1)
        avg_var_copy_f[observed] = float('nan')
        mean_dist_n_copy_f[observed] = float('nan')
        avg_var_mean_f = torch.nanmean(avg_var_copy_f, dim=1)
        mean_dist_n_mean_f = torch.nanmean(mean_dist_n_copy_f, dim=1)

        df = pd.DataFrame({
            'obs': obs,
            'avg_var_all': avg_var_all,
            'avg_var_mean_t': avg_var_mean_t.numpy(),
            'avg_var_mean_f': avg_var_mean_f.numpy()
        })

        df_melted = df.melt(id_vars=['obs'], value_vars=['avg_var_mean_t', 'avg_var_mean_f', 'avg_var_all'],
                            var_name='category', value_name='mean')

        category_mapping = {
            'avg_var_mean_t': 'Observed points',
            'avg_var_mean_f': 'Reconstructed points',
            'avg_var_all': 'All points'
        }
        df_melted['category'] = df_melted['category'].map(category_mapping)
        df_melted = df_melted.dropna(subset=['mean'])

        # df['obs'] = df['obs'].astype(int)
        plt.figure(figsize=(12, 6))

        sns.boxplot(x='obs', y='mean', hue='category', data=df_melted)

        plt.xlabel('Observed portion of the length of the femur [%]')
        plt.ylabel(r'$Average\ variance\ of\ the\ points\ [\mathrm{mm}^{2}]$')
        plt.title('Reconstruction of a previously unseen femur (LOOCV with N=47) using parallel MCMC sampling (20 chains '
                'with 20000 samples each).')

        plt.legend(title='Type of points analysed', bbox_to_anchor=(1.05, 1), loc='upper right')

        plt.ylim(0, 25)

        plt.tight_layout()
        plt.show()

    if READ_IN:
        # Test procedure with subsequent visualisation
        rel_model_path = Path(REL_PATH_MODEL)
        model_path = Path.cwd().parent / rel_model_path
        model_reader = ModelReader(model_path)
        shape = custom_io.read_meshes(REL_PATH_REFERENCE, DEVICE)[0][0]
        model = model_reader.get_model(DEVICE)

        starting_params = torch.zeros(model.rank, device=DEVICE)
        shape.set_points(model.get_points_from_parameters(starting_params), reset_com=True)
        full_target = shape.copy()
        full_target.set_points(model.get_points_from_parameters(1.5 * torch.ones(model.rank, device=DEVICE)), reset_com=True)
        z_min, z_max = torch.min(full_target.tensor_points, dim=0)[1][2].item(), torch.max(full_target.tensor_points, dim=0)[1][2].item()
        part_target = full_target.partial_shape(z_max, z_min, 0.5)
        batched_target = BatchTorchMesh(part_target, 'target', DEVICE, BATCH_SIZE)
        batched_shape = BatchTorchMesh(shape, 'current_shapes', DEVICE, BATCH_SIZE)

        if PROPOSAL_TYPE == "GAUSS_RAND":
            random_walk = GaussianRandomWalkProposal(BATCH_SIZE, starting_params, DEVICE, SIGMA_MOD_GAUSS,
                                                     SIGMA_TRANS_GAUSS, SIGMA_ROT)
        elif PROPOSAL_TYPE == "CP_SIMPLE":
            random_walk = ClosestPointProposal(BATCH_SIZE, starting_params, DEVICE, batched_shape,
                                               batched_target,
                                               model, SIGMA_MOD_CP, SIGMA_TRANS_CP, SIGMA_ROT, CP_D,
                                               CP_RECALCULATION_PERIOD)
        sampler = PDMMetropolisSampler(model, random_walk, batched_shape, batched_target, correspondences=False)

        generator = torch.Generator(device=DEVICE)
        for i in tqdm(range(CHAIN_LENGTH)):
            random = torch.rand(1, device=DEVICE, generator=generator).item()
            if random < MODEL_PROBABILITY:
                proposal = ParameterProposalType.MODEL
            elif MODEL_PROBABILITY <= random < MODEL_PROBABILITY + TRANSLATION_PROBABILITY:
                proposal = ParameterProposalType.TRANSLATION
            else:
                proposal = ParameterProposalType.ROTATION
            sampler.propose(proposal)
            sampler.determine_quality(proposal)
            sampler.decide(proposal, full_target)

        # batched_shape.change_device(torch.device("cpu"))
        # model.change_device(torch.device("cpu"))
        # sampler.change_device(torch.device("cpu"))
        visualizer = MainVisualizer(batched_shape, model, sampler)
        acceptance_ratios = sampler.acceptance_ratio()
        print("Acceptance Ratios:")
        strings = ['Parameters', 'Translation', 'Rotation', 'Total']
        for desc, val in zip(strings, acceptance_ratios):
            print(f"{desc}: {val:.4f}")
        visualizer.run()

    else:
        # LOOCV (Leave-One-Out Cross-Validation) procedure
        meshes, _ = custom_io.read_meshes(REL_PATH_MESH, DEVICE)
        for percentage in PERCENTAGES_OBSERVED_LENGTH:
            for l in range(len(meshes)):
                target = meshes[l]
                # full_target = target.simplify_qem(DECIMATION_TARGET)
                # z_min, z_max = torch.min(full_target.tensor_points, dim=0)[1][2].item(), \
                # torch.max(full_target.tensor_points, dim=0)[1][2].item()
                # part_target = full_target.partial_shape(z_max, z_min, 1 - percentage)
                # dists = distance_to_closest_point(full_target.tensor_points.unsqueeze(-1), part_target.tensor_points, 1)
                # observed = (dists < 1e-6).squeeze()
                del meshes[l]
                model = PointDistributionModel(meshes=meshes)
                # LATER
                # meshes.insert(l, target)
                shape = model.decimate(DECIMATION_TARGET)
                # model.decimate() uses the first mesh of the list as its reference.
                full_target = target.simplify_ref(meshes[0], shape)
                z_min, z_max = torch.min(full_target.tensor_points, dim=0)[1][2].item(), \
                torch.max(full_target.tensor_points, dim=0)[1][2].item()
                part_target = full_target.partial_shape(z_max, z_min, 1 - percentage)
                dists = distance_to_closest_point(full_target.tensor_points.unsqueeze(-1), part_target.tensor_points, 1)
                observed = (dists < 1e-6).squeeze()
                batched_target = BatchTorchMesh(part_target, 'target', DEVICE, BATCH_SIZE)
                starting_params = torch.zeros(model.rank, device=DEVICE)
                shape.set_points(model.get_points_from_parameters(starting_params), reset_com=True)
                batched_shape = BatchTorchMesh(shape, 'current_shapes', DEVICE, BATCH_SIZE)
                if PROPOSAL_TYPE == "GAUSS_RAND":
                    random_walk = GaussianRandomWalkProposal(BATCH_SIZE, starting_params, DEVICE, SIGMA_MOD_GAUSS,
                                                             SIGMA_TRANS_GAUSS, SIGMA_ROT)
                elif PROPOSAL_TYPE == "CP_SIMPLE":
                    random_walk = ClosestPointProposal(BATCH_SIZE, starting_params, DEVICE, batched_shape,
                                                       batched_target,
                                                       model, SIGMA_MOD_CP, SIGMA_TRANS_CP, SIGMA_ROT, CP_D,
                                                       CP_RECALCULATION_PERIOD)
                sampler = PDMMetropolisSampler(model, random_walk, batched_shape, batched_target, correspondences=False,
                                               save_full_mesh_chain=True, save_residuals=True)
                # The fact that the meshes used are already registered can be used here.
                # icp = ICPAnalyser(batched_target, batched_shape, mode=ICPMode.BATCHED)
                # icp.icp()
                generator = torch.Generator(device=DEVICE)
                for i in tqdm(range(CHAIN_LENGTH)):
                    random = torch.rand(1, device=DEVICE, generator=generator).item()
                    if random < MODEL_PROBABILITY:
                        proposal = ParameterProposalType.MODEL
                    elif MODEL_PROBABILITY <= random < MODEL_PROBABILITY + TRANSLATION_PROBABILITY:
                        proposal = ParameterProposalType.TRANSLATION
                    else:
                        proposal = ParameterProposalType.ROTATION
                    sampler.propose(proposal)
                    sampler.determine_quality(proposal)
                    sampler.decide(proposal, full_target)

                # batched_curr_shape.change_device(torch.device("cpu"))
                # model.change_device(torch.device("cpu"))
                # sampler.change_device(torch.device("cpu"))
                # visualizer = MainVisualizer(batched_curr_shape, model, sampler)
                # visualizer.run()

                analyser = ChainAnalyser(sampler, random_walk, model, observed, sampler.full_chain)
                # analyser.detect_burn_in()
                data = analyser.data_to_json(l, int(100 * percentage))
                meshes.insert(l, target)
                # TODO: Write proper output writer class.
                output_dir = Path.cwd().parent / Path(REL_OUTPUT_DIR)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_filename = f'output_data_{l}_{int(100 * percentage)}.json'
                output_file = output_dir / output_filename
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=4)


if __name__ == "__main__":
    # torch.set_default_tensor_type(torch.cuda.FloatTensor)
    run()
    # batched_reference.change_device(torch.device("cpu"))
    # model.change_device(torch.device("cpu"))
    # sampler.change_device(torch.device("cpu"))
    # visualizer = MainVisualizer(batched_reference, model, sampler)
    # acceptance_ratios = sampler.acceptance_ratio()
    # print("Acceptance Ratios:")
    # strings = ['Parameters', 'Translation', 'Rotation', 'Total']
    # for desc, val in zip(strings, acceptance_ratios):
    #    print(f"{desc}: {val:.4f}")
    # visualizer.run()
