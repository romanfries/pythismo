import pandas as pd
import json
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt

import torch
from torch.nn.utils.rnn import pad_sequence


class DataHandler:
    def __init__(self, rel_dir='datasets/output'):
        self.class_dir = Path(__file__).parent.parent.parent
        self.main_dir = self.class_dir / Path(rel_dir)
        # self.statistics_dir = self.main_dir / Path('statistics')
        self.statistics_dir = self.main_dir
        self.chain_dir = self.main_dir / Path('chains')
        self.plot_dir = self.main_dir / Path('plots')
        self.statistics_dir.mkdir(parents=True, exist_ok=True)
        self.chain_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    def write_statistics(self, data, loo, obs, additional_param):
        output_filename = f'mcmc_{loo}_{obs}_{additional_param}.json'
        output_file = self.statistics_dir / output_filename
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)

    def read_all_statistics(self):
        ess_list = []
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

                ess = torch.tensor(loaded_data['effective_sample_sizes']['ess_per_param'])
                ess_list.append(ess)

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
                vars_ = torch.tensor(loaded_data['unnormalised_log_density_posterior']['var'])

                means_list.append(means)
                mins_list.append(mins)
                maxs_list.append(maxs)
                vars_list.append(vars_)

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
            'effective_sample_sizes': pad_sequence(ess_list, batch_first=True),
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

            ess = torch.tensor(loaded_data['effective_sample_sizes']['ess_per_param'])

            mean_dist_c_post = torch.tensor(loaded_data['accuracy']['mean_dist_corr_post'])
            mean_dist_n_post = torch.tensor(loaded_data['accuracy']['mean_dist_clp_post'])
            mean_dist_c_map = torch.tensor(loaded_data['accuracy']['mean_dist_corr_map'])
            mean_dist_n_map = torch.tensor(loaded_data['accuracy']['mean_dist_clp_map'])
            avg_var_post = torch.tensor(loaded_data['accuracy']['var_post'])
            avg_var_map = torch.tensor(loaded_data['accuracy']['var_map'])

            means = torch.tensor(loaded_data['unnormalised_log_density_posterior']['mean'])
            mins = torch.tensor(loaded_data['unnormalised_log_density_posterior']['min'])
            maxs = torch.tensor(loaded_data['unnormalised_log_density_posterior']['max'])
            vars_ = torch.tensor(loaded_data['unnormalised_log_density_posterior']['var'])

            observed = torch.tensor(loaded_data['observed']['boolean'], dtype=torch.bool)

            acc_par = loaded_data['acceptance']['model']
            acc_trans = loaded_data['acceptance']['translation']
            acc_rot = loaded_data['acceptance']['rotation']
            acc_tot = loaded_data['acceptance']['total']

            loo = loaded_data['identifiers']['reconstructed_shape']
            obs = loaded_data['identifiers']['percentage_observed']

        data = {
            'effective_sample_sizes': ess,
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
                'var': vars_
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

    def write_chain_and_residuals(self, dict_chain_and_residuals, loo, obs):
        # There is currently no application implemented for stored chains and residuals.
        output_filename = f'mesh_chain_{loo}_{obs}.pt'
        output_file = self.chain_dir / output_filename
        torch.save(dict_chain_and_residuals, output_file)

    def write_param_chain_posterior(self, dict_param_chain_posterior, loo, obs):
        # There is currently no application implemented for stored parameter chains.
        output_filename = f'param_chain_{loo}_{obs}.pt'
        output_file = self.chain_dir / output_filename
        torch.save(dict_param_chain_posterior, output_file)

    def generate_plots(self, data_dict=None):
        # Needs to be adjusted depending on the experiments performed
        if data_dict is None:
            data_dict = self.read_all_statistics()

        # Prepare the data
        mean_dist_c_post = data_dict['accuracy']['mean_dist_corr_post']
        mean_dist_n_post = data_dict['accuracy']['mean_dist_clp_post']
        mean_dist_c_map = data_dict['accuracy']['mean_dist_corr_map']
        mean_dist_n_map = data_dict['accuracy']['mean_dist_clp_map']
        avg_var_post = data_dict['accuracy']['var_post']
        avg_var_map = data_dict['accuracy']['var_map']

        obs = data_dict['identifiers']['percentage_observed']
        observed = data_dict['observed']

        # avg_var_all = avg_var.mean(dim=1).numpy()
        mean_dist_c_post_all = mean_dist_c_post.mean(dim=1).numpy()
        mean_dist_n_post_all = mean_dist_n_post.mean(dim=1).numpy()
        mean_dist_c_map_all = mean_dist_c_map.mean(dim=1).numpy()
        mean_dist_n_map_all = mean_dist_n_map.mean(dim=1).numpy()
        avg_var_post_all = avg_var_post.mean(dim=1).numpy()
        avg_var_map_all = avg_var_map.mean(dim=1).numpy()

        # avg_var_copy_t, avg_var_copy_f = avg_var.clone(), avg_var.clone()
        mean_dist_c_post_copy_t, mean_dist_c_post_copy_f = mean_dist_c_post.clone(), mean_dist_c_post.clone()
        mean_dist_n_post_copy_t, mean_dist_n_post_copy_f = mean_dist_n_post.clone(), mean_dist_n_post.clone()
        mean_dist_c_map_copy_t, mean_dist_c_map_copy_f = mean_dist_c_map.clone(), mean_dist_c_map.clone()
        mean_dist_n_map_copy_t, mean_dist_n_map_copy_f = mean_dist_n_map.clone(), mean_dist_n_map.clone()
        avg_var_post_copy_t, avg_var_post_copy_f = avg_var_post.clone(), avg_var_post.clone()
        avg_var_map_copy_t, avg_var_map_copy_f = avg_var_map.clone(), avg_var_map.clone()

        # avg_var_copy_t[~obs] = float('nan')
        mean_dist_c_post_copy_t[~observed] = float('nan')
        mean_dist_n_post_copy_t[~observed] = float('nan')
        mean_dist_c_map_copy_t[~observed] = float('nan')
        mean_dist_n_map_copy_t[~observed] = float('nan')
        avg_var_post_copy_t[~observed] = float('nan')
        avg_var_map_copy_t[~observed] = float('nan')

        # avg_var_mean_t = torch.nanmean(avg_var_copy_t, dim=1)
        mean_dist_c_post_mean_t = torch.nanmean(mean_dist_c_post_copy_t, dim=1)
        mean_dist_n_post_mean_t = torch.nanmean(mean_dist_n_post_copy_t, dim=1)
        mean_dist_c_map_mean_t = torch.nanmean(mean_dist_c_map_copy_t, dim=1)
        mean_dist_n_map_mean_t = torch.nanmean(mean_dist_n_map_copy_t, dim=1)
        avg_var_post_mean_t = torch.nanmean(avg_var_post_copy_t, dim=1)
        avg_var_map_mean_t = torch.nanmean(avg_var_map_copy_t, dim=1)

        # avg_var_copy_f[obs] = float('nan')
        mean_dist_c_post_copy_f[observed] = float('nan')
        mean_dist_n_post_copy_f[observed] = float('nan')
        mean_dist_c_map_copy_f[observed] = float('nan')
        mean_dist_n_map_copy_f[observed] = float('nan')
        avg_var_post_copy_f[observed] = float('nan')
        avg_var_map_copy_f[observed] = float('nan')

        # avg_var_mean_f = torch.nanmean(avg_var_copy_f, dim=1)
        mean_dist_c_post_mean_f = torch.nanmean(mean_dist_c_post_copy_f, dim=1)
        mean_dist_n_post_mean_f = torch.nanmean(mean_dist_n_post_copy_f, dim=1)
        mean_dist_c_map_mean_f = torch.nanmean(mean_dist_c_map_copy_f, dim=1)
        mean_dist_n_map_mean_f = torch.nanmean(mean_dist_n_map_copy_f, dim=1)
        avg_var_post_mean_f = torch.nanmean(avg_var_post_copy_f, dim=1)
        avg_var_map_mean_f = torch.nanmean(avg_var_map_copy_f, dim=1)

        # Average distance to corresponding point across all samples
        df = pd.DataFrame({
            'obs': obs.numpy(),
            'mean_dist_c_post_all': mean_dist_c_post_all,
            'mean_dist_c_post_mean_t': mean_dist_c_post_mean_t.numpy(),
            'mean_dist_c_post_mean_f': mean_dist_c_post_mean_f.numpy()
        })

        df_melted = df.melt(id_vars=['obs'],
                            value_vars=['mean_dist_c_post_all', 'mean_dist_c_post_mean_t', 'mean_dist_c_post_mean_f'],
                            var_name='category', value_name='mean')

        category_mapping = {
            'mean_dist_c_post_all': 'All points',
            'mean_dist_c_post_mean_t': 'Observed points',
            'mean_dist_c_post_mean_f': 'Reconstructed points'
        }
        df_melted['category'] = df_melted['category'].map(category_mapping)
        df_melted = df_melted.dropna(subset=['mean'])

        # df['obs'] = df['obs'].astype(int)
        plt.figure(figsize=(20, 10), dpi=300)

        sns.boxplot(x='obs', y='mean', hue='category', data=df_melted)

        plt.xlabel('Observed portion of the length of the femur [%]')
        plt.ylabel(r'$Average\ distance\ to\ the\ corresponding\ point\ on\ the\ reconstruction\ surface\ (utilising\ '
                   r'given\ correspondences)\ [\mathrm{mm}]$')
        plt.title(
            'Femur reconstruction (LOOCV with N=47) using parallel MCMC sampling (20 chains with 20000 samples each)')
        plt.legend(title='Type of points analysed', bbox_to_anchor=(1.05, 1), loc='upper right')
        plt.ylim(0, 40)
        plt.tight_layout()
        filename = "post_dist_corr_split.png"
        png_plot_file = self.plot_dir / filename
        plt.savefig(png_plot_file)
        plt.close()

        # Average distance to the closest point across all samples
        df = pd.DataFrame({
            'obs': obs.numpy(),
            'mean_dist_n_post_all': mean_dist_n_post_all,
            'mean_dist_n_post_mean_t': mean_dist_n_post_mean_t.numpy(),
            'mean_dist_n_post_mean_f': mean_dist_n_post_mean_f.numpy()
        })

        df_melted = df.melt(id_vars=['obs'],
                            value_vars=['mean_dist_n_post_all', 'mean_dist_n_post_mean_t', 'mean_dist_n_post_mean_f'],
                            var_name='category', value_name='mean')

        category_mapping = {
            'mean_dist_n_post_all': 'All points',
            'mean_dist_n_post_mean_t': 'Observed points',
            'mean_dist_n_post_mean_f': 'Reconstructed points'
        }
        df_melted['category'] = df_melted['category'].map(category_mapping)
        df_melted = df_melted.dropna(subset=['mean'])

        # df['obs'] = df['obs'].astype(int)
        plt.figure(figsize=(20, 10), dpi=300)

        sns.boxplot(x='obs', y='mean', hue='category', data=df_melted)

        plt.xlabel('Observed portion of the length of the femur [%]')
        plt.ylabel(r'$Average\ distance\ to\ the\ closest\ point\ on\ the\ reconstructed\ surface\ [\mathrm{mm}]$')
        plt.title(
            'Femur reconstruction (LOOCV with N=47) using parallel MCMC sampling (20 chains with 20000 samples each)')
        plt.legend(title='Type of points analysed', bbox_to_anchor=(1.05, 1), loc='upper right')
        plt.ylim(0, 40)
        plt.tight_layout()
        filename = "post_dist_clp_split.png"
        png_plot_file = self.plot_dir / filename
        plt.savefig(png_plot_file)
        plt.close()

        # Average distance to the corresponding point across the MAP estimates of all MCMC chains
        df = pd.DataFrame({
            'obs': obs.numpy(),
            'mean_dist_c_map_all': mean_dist_c_map_all,
            'mean_dist_c_map_mean_t': mean_dist_c_map_mean_t.numpy(),
            'mean_dist_c_map_mean_f': mean_dist_c_map_mean_f.numpy()
        })

        df_melted = df.melt(id_vars=['obs'],
                            value_vars=['mean_dist_c_map_all', 'mean_dist_c_map_mean_t', 'mean_dist_c_map_mean_f'],
                            var_name='category', value_name='mean')

        category_mapping = {
            'mean_dist_c_map_all': 'All points',
            'mean_dist_c_map_mean_t': 'Observed points',
            'mean_dist_c_map_mean_f': 'Reconstructed points'
        }
        df_melted['category'] = df_melted['category'].map(category_mapping)
        df_melted = df_melted.dropna(subset=['mean'])

        # df['obs'] = df['obs'].astype(int)
        plt.figure(figsize=(20, 10), dpi=300)

        sns.boxplot(x='obs', y='mean', hue='category', data=df_melted)

        plt.xlabel('Observed portion of the length of the femur [%]')
        plt.ylabel(
            r'$Average\ distance\ to\ the\ corresponding\ point\ on\ the\ reconstructed\ surface\ [\mathrm{mm}]$')
        plt.title(
            'Femur reconstruction (LOOCV with N=47) using parallel MCMC sampling (20 chains, MAP sample selected for'
            'each one)')
        plt.legend(title='Type of points analysed', bbox_to_anchor=(1.05, 1), loc='upper right')
        plt.ylim(0, 40)
        plt.tight_layout()
        filename = "map_dist_corr_split.png"
        png_plot_file = self.plot_dir / filename
        plt.savefig(png_plot_file)
        plt.close()

        # Average distance to the closest point across the MAP estimates of all MCMC chains
        df = pd.DataFrame({
            'obs': obs.numpy(),
            'mean_dist_n_map_all': mean_dist_n_map_all,
            'mean_dist_n_map_mean_t': mean_dist_n_map_mean_t.numpy(),
            'mean_dist_n_map_mean_f': mean_dist_n_map_mean_f.numpy()
        })

        df_melted = df.melt(id_vars=['obs'],
                            value_vars=['mean_dist_n_map_all', 'mean_dist_n_map_mean_t', 'mean_dist_n_map_mean_f'],
                            var_name='category', value_name='mean')

        category_mapping = {
            'mean_dist_n_map_all': 'All points',
            'mean_dist_n_map_mean_t': 'Observed points',
            'mean_dist_n_map_mean_f': 'Reconstructed points'
        }
        df_melted['category'] = df_melted['category'].map(category_mapping)
        df_melted = df_melted.dropna(subset=['mean'])

        # df['obs'] = df['obs'].astype(int)
        plt.figure(figsize=(20, 10), dpi=300)

        sns.boxplot(x='obs', y='mean', hue='category', data=df_melted)

        plt.xlabel('Observed portion of the length of the femur [%]')
        plt.ylabel(
            r'$Average\ distance\ to\ the\ closest\ point\ on\ the\ reconstructed\ surface\ [\mathrm{mm}]$')
        plt.title(
            'Femur reconstruction (LOOCV with N=47) using parallel MCMC sampling (20 chains, MAP sample selected for'
            'each one)')
        plt.legend(title='Type of points analysed', bbox_to_anchor=(1.05, 1), loc='upper right')
        plt.ylim(0, 40)
        plt.tight_layout()
        filename = "map_dist_clp_split.png"
        png_plot_file = self.plot_dir / filename
        plt.savefig(png_plot_file)
        plt.close()

        # Average variance of the points across all chains and samples
        df = pd.DataFrame({
            'obs': obs.numpy(),
            'avg_var_post_all': avg_var_post_all,
            'avg_var_post_mean_t': avg_var_post_mean_t.numpy(),
            'avg_var_post_mean_f': avg_var_post_mean_f.numpy()
        })

        df_melted = df.melt(id_vars=['obs'],
                            value_vars=['avg_var_post_all', 'avg_var_post_mean_t', 'avg_var_post_mean_f'],
                            var_name='category', value_name='mean')

        category_mapping = {
            'avg_var_post_all': 'All points',
            'avg_var_post_mean_t': 'Observed points',
            'avg_var_post_mean_f': 'Reconstructed points'
        }
        df_melted['category'] = df_melted['category'].map(category_mapping)
        df_melted = df_melted.dropna(subset=['mean'])

        # df['obs'] = df['obs'].astype(int)
        plt.figure(figsize=(20, 10), dpi=300)

        sns.boxplot(x='obs', y='mean', hue='category', data=df_melted)

        plt.xlabel('Observed portion of the length of the femur [%]')
        plt.ylabel(
            r'$Average\ variance\ of\ the\ points\ [\mathrm{mm}^{2}]$')
        plt.title(
            'Femur reconstruction (LOOCV with N=47) using parallel MCMC sampling (20 chains with 20000 samples each)')
        plt.legend(title='Type of points analysed', bbox_to_anchor=(1.05, 1), loc='upper right')
        plt.ylim(0, 100)
        plt.tight_layout()
        filename = "post_var_split.png"
        png_plot_file = self.plot_dir / filename
        plt.savefig(png_plot_file)
        plt.close()

        # Average variance of the points across the MAP estimates of all chains
        df = pd.DataFrame({
            'obs': obs.numpy(),
            'avg_var_map_all': avg_var_map_all,
            'avg_var_map_mean_t': avg_var_map_mean_t.numpy(),
            'avg_var_map_mean_f': avg_var_map_mean_f.numpy()
        })

        df_melted = df.melt(id_vars=['obs'],
                            value_vars=['avg_var_map_all', 'avg_var_map_mean_t', 'avg_var_map_mean_f'],
                            var_name='category', value_name='mean')

        category_mapping = {
            'avg_var_map_all': 'All points',
            'avg_var_map_mean_t': 'Observed points',
            'avg_var_map_mean_f': 'Reconstructed points'
        }
        df_melted['category'] = df_melted['category'].map(category_mapping)
        df_melted = df_melted.dropna(subset=['mean'])

        # df['obs'] = df['obs'].astype(int)
        plt.figure(figsize=(20, 10), dpi=300)

        sns.boxplot(x='obs', y='mean', hue='category', data=df_melted)

        plt.xlabel('Observed portion of the length of the femur [%]')
        plt.ylabel(
            r'$Average\ variance\ of\ the\ points\ [\mathrm{mm}^{2}]$')
        plt.title(
            'Femur reconstruction (LOOCV with N=47) using parallel MCMC sampling (20 chains, MAP sample selected for'
            'each one)')
        plt.legend(title='Type of points analysed', bbox_to_anchor=(1.05, 1), loc='upper right')
        plt.ylim(0, 100)
        plt.tight_layout()
        filename = "map_var_split.png"
        png_plot_file = self.plot_dir / filename
        plt.savefig(png_plot_file)
        plt.close()

        # Average distance to corresponding point across all samples
        df = pd.DataFrame({
            'obs': obs.numpy(),
            'mean_dist_c_post_all': mean_dist_c_post_all
        })

        plt.figure(figsize=(20, 10))
        sns.boxplot(x='obs', y='mean_dist_c_post_all', data=df, color='skyblue')

        plt.xlabel('Observed portion of the length of the femur [%]')
        plt.ylabel(r'$Average\ distance\ to\ the\ corresponding\ point\ on\ the\ reconstruction\ surface\ (utilising\ '
                   r'given\ correspondences)\ [\mathrm{mm}]$')
        plt.title(
            'Femur reconstruction (LOOCV with N=47) using parallel MCMC sampling (20 chains with 20000 samples each)')
        plt.ylim(0, 25)
        plt.tight_layout()
        filename = "post_dist_corr_all.png"
        png_plot_file = self.plot_dir / filename
        plt.savefig(png_plot_file)
        plt.close()

        # Average distance to the closest point across all samples
        df = pd.DataFrame({
            'obs': obs.numpy(),
            'mean_dist_n_post_all': mean_dist_n_post_all
        })

        plt.figure(figsize=(20, 10))
        sns.boxplot(x='obs', y='mean_dist_n_post_all', data=df, color='skyblue')

        plt.xlabel('Observed portion of the length of the femur [%]')
        plt.ylabel(r'$Average\ distance\ to\ the\ closest\ point\ on\ the\ reconstructed\ surface\ [\mathrm{mm}]$')
        plt.title(
            'Femur reconstruction (LOOCV with N=47) using parallel MCMC sampling (20 chains with 20000 samples each)')
        plt.ylim(0, 25)
        plt.tight_layout()
        filename = "post_dist_clp_all.png"
        png_plot_file = self.plot_dir / filename
        plt.savefig(png_plot_file)
        plt.close()

        # Average distance to the corresponding point across the MAP estimates of all MCMC chains
        df = pd.DataFrame({
            'obs': obs.numpy(),
            'mean_dist_c_map_all': mean_dist_c_map_all
        })

        plt.figure(figsize=(20, 10))
        sns.boxplot(x='obs', y='mean_dist_c_map_all', data=df, color='skyblue')

        plt.xlabel('Observed portion of the length of the femur [%]')
        plt.ylabel(r'$Average\ distance\ to\ the\ corresponding\ point\ on\ the\ reconstruction\ surface\ (utilising\ '
                   r'given\ correspondences)\ [\mathrm{mm}]$')
        plt.title(
            'Femur reconstruction (LOOCV with N=47) using parallel MCMC sampling (20 chains, MAP sample selected for'
            'each one)')
        plt.ylim(0, 25)
        plt.tight_layout()
        filename = "map_dist_corr_all.png"
        png_plot_file = self.plot_dir / filename
        plt.savefig(png_plot_file)
        plt.close()

        # Average distance to the closest point across the MAP estimates of all MCMC chains
        df = pd.DataFrame({
            'obs': obs.numpy(),
            'mean_dist_n_map_all': mean_dist_n_map_all
        })

        plt.figure(figsize=(20, 10))
        sns.boxplot(x='obs', y='mean_dist_n_map_all', data=df, color='skyblue')

        plt.xlabel('Observed portion of the length of the femur [%]')
        plt.ylabel(r'$Average\ distance\ to\ the\ closest\ point\ on\ the\ reconstructed\ surface\ [\mathrm{mm}]$')
        plt.title(
            'Femur reconstruction (LOOCV with N=47) using parallel MCMC sampling (20 chains, MAP sample selected for'
            'each one)')
        plt.ylim(0, 25)
        plt.tight_layout()
        filename = "map_dist_clp_all.png"
        png_plot_file = self.plot_dir / filename
        plt.savefig(png_plot_file)
        plt.close()

        # Average variance of the points across all chains and samples
        df = pd.DataFrame({
            'obs': obs.numpy(),
            'avg_var_post_all': avg_var_post_all
        })

        plt.figure(figsize=(20, 10))
        sns.boxplot(x='obs', y='avg_var_post_all', data=df, color='skyblue')

        plt.xlabel('Observed portion of the length of the femur [%]')
        plt.ylabel(r'$Average\ variance\ of\ the\ points\ [\mathrm{mm}^{2}]$')
        plt.title(
            'Femur reconstruction (LOOCV with N=47) using parallel MCMC sampling (20 chains with 20000 samples each)')
        plt.ylim(0, 80)
        plt.tight_layout()
        filename = "post_var_all.png"
        png_plot_file = self.plot_dir / filename
        plt.savefig(png_plot_file)
        plt.close()

        # Average variance of the points across the MAP estimates of all chains
        df = pd.DataFrame({
            'obs': obs.numpy(),
            'avg_var_map_all': avg_var_map_all
        })

        plt.figure(figsize=(20, 10))
        sns.boxplot(x='obs', y='avg_var_map_all', data=df, color='skyblue')

        plt.xlabel('Observed portion of the length of the femur [%]')
        plt.ylabel(r'$Average\ variance\ of\ the\ points\ [\mathrm{mm}^{2}]$')
        plt.title(
            'Femur reconstruction (LOOCV with N=47) using parallel MCMC sampling (20 chains, MAP sample selected for'
            'each one)')
        plt.ylim(0, 80)
        plt.tight_layout()
        filename = "map_var_all.png"
        png_plot_file = self.plot_dir / filename
        plt.savefig(png_plot_file)
        plt.close()
