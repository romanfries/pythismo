import torch
import json
from pathlib import Path


class DataHandler:
    def __init__(self, rel_dir='datasets/output'):
        self.class_dir = Path(__file__).parent.parent.parent
        self.main_dir = self.class_dir / Path(rel_dir)
        self.statistics_dir = self.main_dir / Path('statistics')
        self.chain_dir = self.main_dir / Path('chains')
        self.statistics_dir.mkdir(parents=True, exist_ok=True)
        self.chain_dir.mkdir(parents=True, exist_ok=True)

    def write_statistics(self, data, loo, obs):
        output_filename = f'mcmc_{loo}_{obs}.json'
        output_file = self.statistics_dir / output_filename
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)

    def read_all_statistics(self):
        mean_dist_c_post_list, mean_dist_n_post_list, mean_dist_c_map_list, mean_dist_n_map_list = [], [], [], []
        avg_var_post_list, avg_var_map_list = [], []
        means_list, mins_list, maxs_list, vars_list = [], [], [], []
        observed_list = []
        acc_par_list, acc_trans_list, acc_rot_list, acc_tot_list = [], [], [], []
        loo_list, obs_list = [], []

        json_files = sorted(self.statistics_dir.glob('*.json'))

        for json_file in json_files:
            with open(json_file, 'r') as f:
                loaded_data = json.load(f)

                mean_dist_c_post = torch.tensor(loaded_data['accuracy']['mean_dist_corr_post'])
                mean_dist_n_post = torch.tensor(loaded_data['accuracy']['mean_dist_clp_post'])
                mean_dist_c_map = torch.tensor(loaded_data['accuracy']['mean_dist_corr_map'])
                mean_dist_n_map = torch.tensor(loaded_data['accuracy']['mean_dist_clp_map'])
                avg_var_post = torch.tensor(loaded_data['accuracy']['var_post'])
                avg_var_map = torch.tensor(loaded_data['accuracy']['var_map'])

                mean_dist_c_post_list.append(mean_dist_c_post)
                mean_dist_n_post_list.append(mean_dist_n_post)
                mean_dist_c_map_list.append(mean_dist_c_map)
                mean_dist_n_map_list.append(mean_dist_n_map)
                avg_var_post_list.append(avg_var_post)
                avg_var_map_list.append(avg_var_map)

                means = torch.tensor(loaded_data['unnormalised_log_density_posterior']['mean'])
                mins = torch.tensor(loaded_data['unnormalised_log_density_posterior']['min'])
                maxs = torch.tensor(loaded_data['unnormalised_log_density_posterior']['max'])
                vars = torch.tensor(loaded_data['unnormalised_log_density_posterior']['var'])

                means_list.append(means)
                mins_list.append(mins)
                maxs_list.append(maxs)
                vars_list.append(vars)

                observed = torch.tensor(loaded_data['observed']['boolean'], dtype=torch.bool)
                observed_list.append(observed)

                acc_par = loaded_data['acceptance']['model']
                acc_trans = loaded_data['acceptance']['translation']
                acc_rot = loaded_data['acceptance']['rotation']
                acc_tot = loaded_data['acceptance']['total']

                acc_par_list.append(acc_par)
                acc_trans_list.append(acc_trans)
                acc_rot_list.append(acc_rot)
                acc_tot_list.append(acc_tot)

                loo = loaded_data['identifiers']['reconstructed_shape']
                obs = loaded_data['identifiers']['percentage_observed']

                loo_list.append(loo)
                obs_list.append(obs)

        stacked_data = {
            'accuracy': {
                'mean_dist_corr_post': torch.stack(mean_dist_c_post_list),
                'mean_dist_clp_post': torch.stack(mean_dist_n_post_list),
                'mean_dist_corr_map': torch.stack(mean_dist_c_map_list),
                'mean_dist_clp_map': torch.stack(mean_dist_n_map_list),
                'var_post': torch.stack(avg_var_post_list),
                'var_map': torch.stack(avg_var_map_list)
            },
            'unnormalised_log_density_posterior': {
                'mean': torch.stack(means_list),
                'min': torch.stack(mins_list),
                'max': torch.stack(maxs_list),
                'var': torch.stack(vars_list)
            },
            'observed': torch.stack(observed_list),
            'acceptance': {
                'model': torch.tensor(acc_par_list),
                'translation': torch.tensor(acc_trans_list),
                'rotation': torch.tensor(acc_rot_list),
                'total': torch.tensor(acc_tot_list)
            },
            'identifiers': {
                'reconstructed_shape': torch.tensor(loo_list),
                'percentage_observed': torch.tensor(obs_list)
            }
        }

        return stacked_data

    def read_single_statistics(self, loo, obs):
        input_filename = f'mcmc_{loo}_{obs}.json'
        input_file = self.statistics_dir / input_filename
        with open(input_file, 'r') as f:
            loaded_data = json.load(f)

            mean_dist_c_post = torch.tensor(loaded_data['accuracy']['mean_dist_corr_post'])
            mean_dist_n_post = torch.tensor(loaded_data['accuracy']['mean_dist_clp_post'])
            mean_dist_c_map = torch.tensor(loaded_data['accuracy']['mean_dist_corr_map'])
            mean_dist_n_map = torch.tensor(loaded_data['accuracy']['mean_dist_clp_map'])
            avg_var_post = torch.tensor(loaded_data['accuracy']['var_post'])
            avg_var_map = torch.tensor(loaded_data['accuracy']['var_map'])

            means = torch.tensor(loaded_data['unnormalised_log_density_posterior']['mean'])
            mins = torch.tensor(loaded_data['unnormalised_log_density_posterior']['min'])
            maxs = torch.tensor(loaded_data['unnormalised_log_density_posterior']['max'])
            vars = torch.tensor(loaded_data['unnormalised_log_density_posterior']['var'])

            observed = torch.tensor(loaded_data['observed']['boolean'], dtype=torch.bool)

            acc_par = loaded_data['acceptance']['model']
            acc_trans = loaded_data['acceptance']['translation']
            acc_rot = loaded_data['acceptance']['rotation']
            acc_tot = loaded_data['acceptance']['total']

            loo = loaded_data['identifiers']['reconstructed_shape']
            obs = loaded_data['identifiers']['percentage_observed']

        data = {
            'accuracy': {
                'mean_dist_corr_post': mean_dist_c_post,
                'mean_dist_clp_post': mean_dist_n_post,
                'mean_dist_corr_map': mean_dist_c_map,
                'mean_dist_clp_map': mean_dist_n_map,
                'var_post': avg_var_post,
                'var_map': avg_var_map
            },
            'unnormalised_log_density_posterior': {
                'mean': means,
                'min': mins,
                'max': maxs,
                'var': vars
            },
            'observed': observed,
            'acceptance': {
                'model': acc_par,
                'translation': acc_trans,
                'rotation': acc_rot,
                'total': acc_tot
            },
            'identifiers': {
                'reconstructed_shape': loo,
                'percentage_observed': obs
            }
        }

        return data

    def write_chain_and_residuals(self, chain_and_residuals, loo, obs):
        output_filename = f'chain_{loo}_{obs}.pt'
        output_file = self.chain_dir / output_filename
        torch.save(chain_and_residuals, output_file)
