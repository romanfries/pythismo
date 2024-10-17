import pandas as pd
import json
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt

import torch
from torch.nn.utils.rnn import pad_sequence

from pytorch3d.structures import Meshes
from pytorch3d.vis import plot_scene
from pytorch3d.vis.plotly_vis import AxisArgs


class DataHandler:
    def __init__(self, rel_dir='datasets/output'):
        self.class_dir = Path(__file__).parent.parent.parent
        self.main_dir = self.class_dir / Path(rel_dir)
        self.statistics_dir = self.main_dir / Path('statistics')
        # self.statistics_dir = self.main_dir
        self.chain_dir = self.main_dir / Path('chains')
        self.plot_dir = self.main_dir / Path('plots')
        self.samples_dir = self.main_dir / Path('samples')
        self.statistics_dir.mkdir(parents=True, exist_ok=True)
        self.chain_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)

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
        # acc_par_list, acc_rnd_list, acc_trans_list, acc_rot_list, acc_tot_list = [], [], [], [], []
        acc_par_list, acc_trans_list, acc_rot_list, acc_tot_list = [], [], [], []
        loo_list, obs_list, additional_param_list = [], [], []

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
                # acc_rnd = loaded_data['acceptance']['random_noise']
                acc_trans = loaded_data['acceptance']['translation']
                acc_rot = loaded_data['acceptance']['rotation']
                acc_tot = loaded_data['acceptance']['total']

                acc_par_list.append(acc_par)
                # acc_rnd_list.append(acc_rnd)
                acc_trans_list.append(acc_trans)
                acc_rot_list.append(acc_rot)
                acc_tot_list.append(acc_tot)

                loo = loaded_data['identifiers']['reconstructed_shape']
                obs = loaded_data['identifiers']['percentage_observed']
                additional_param = loaded_data['identifiers']['additional_param']

                loo_list.append(loo)
                obs_list.append(obs)
                additional_param_list.append(additional_param)

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
                # 'random_noise': torch.tensor(acc_rnd_list),
                'translation': torch.tensor(acc_trans_list),
                'rotation': torch.tensor(acc_rot_list),
                'total': torch.tensor(acc_tot_list)
            },
            'identifiers': {
                'reconstructed_shape': torch.tensor(loo_list),
                'percentage_observed': torch.tensor(obs_list),
                'additional_param': torch.tensor(additional_param_list)
            }
        }

        return stacked_data

    def read_single_statistics(self, loo, obs, additional_param):
        input_filename = f'mcmc_{loo}_{obs}_{additional_param}.json'
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
            acc_rnd = loaded_data['acceptance']['random_noise']
            acc_trans = loaded_data['acceptance']['translation']
            acc_rot = loaded_data['acceptance']['rotation']
            acc_tot = loaded_data['acceptance']['total']

            loo = loaded_data['identifiers']['reconstructed_shape']
            obs = loaded_data['identifiers']['percentage_observed']
            additional_param = loaded_data['identifiers']['additional_param']

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
                'random_noise': acc_rnd,
                'translation': acc_trans,
                'rotation': acc_rot,
                'total': acc_tot
            },
            'identifiers': {
                'reconstructed_shape': loo,
                'percentage_observed': obs,
                'additional_param': additional_param
            }
        }

        return data

    def save_posterior_samples(self, chain, mesh, target, num_samples, loo, obs, additional_param, save_html=False):
        num_points = mesh.num_points
        num_faces = mesh.cells[0].data.size(0)
        _, _, batch_size, chain_length = chain.shape

        # bugged
        # selected = torch.randperm(batch_size * chain_length)[:num_samples]
        selected = torch.multinomial(torch.ones((batch_size * chain_length,), device=chain.device), num_samples)
        verts = chain[:, :, selected // chain_length, selected % chain_length].permute(2, 0, 1)
        faces = mesh.cells[0].data.unsqueeze(0).repeat(num_samples, 1, 1)

        verts_padding = torch.zeros(num_points - target.tensor_points.size(0), 3, device=target.dev)
        verts_padded = torch.cat([target.tensor_points, verts_padding], dim=0)
        faces_padding = - torch.ones(num_faces - target.cells[0].data.size(0), 3, device=target.dev)
        faces_padded = torch.cat([target.cells[0].data, faces_padding], dim=0)

        verts = torch.cat([verts, verts_padded.unsqueeze(0)], dim=0)
        faces = torch.cat([faces, faces_padded.unsqueeze(0)], dim=0)
        meshes = Meshes(verts.to('cpu'), faces.to('cpu'))

        # R, T = look_at_view_transform(2.7, 0, 0)  # 2 camera angles, front and back
        # camera = FoVPerspectiveCameras(device='cpu', R=R, T=T)

        meshes_dict = {
            f"{num_samples} posterior samples of the femur {loo} observed to {obs} per cent": {f"trace_{i}": meshes[i]
                                                                                               for i in
                                                                                               range(num_samples + 1)}}
        # plot mesh batch in the same trace
        fig = plot_scene(meshes_dict, xaxis={"backgroundcolor": "rgb(250, 235, 235)"},
                         yaxis={"backgroundcolor": "rgb(250, 235, 235)"},
                         zaxis={"backgroundcolor": "rgb(250, 235, 235)"}, axis_args=AxisArgs(showgrid=True))

        for idx, mesh_ in enumerate(fig.data):
            mesh_.opacity = 0.90
            mesh_.color = 'limegreen'
        fig.data[-1].opacity = 0.90
        fig.data[-1].color = 'darkmagenta'

        fig.update_annotations(font=dict(size=10))

        if save_html:
            output_filename = f'posterior_samples_{loo}_{obs}_{additional_param}.html'
        else:
            output_filename = f'posterior_samples_{loo}_{obs}_{additional_param}.png'
        output_file = self.samples_dir / output_filename

        if save_html:
            fig.write_html(output_file)
        else:
            fig.write_image(output_file, width=1920, height=1080)

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

    def generate_plots(self, data_dict=None, add_param_available=False):
        # Needs to be adjusted depending on the experiments performed
        if data_dict is None:
            data_dict = self.read_all_statistics()

        # Remove femurs whose Markov chains often do not converge
        # mask = ~torch.isin(data_dict['identifiers']['reconstructed_shape'], torch.tensor([22, 32, 41, 44]))
        # mask = ~torch.gt(torch.var(data_dict['unnormalised_log_density_posterior']['mean'], dim=1), 1.0)
        # Prepare the data
        mean_dist_c_post = data_dict['accuracy']['mean_dist_corr_post']
        mean_dist_n_post = data_dict['accuracy']['mean_dist_clp_post']
        mean_dist_c_map = data_dict['accuracy']['mean_dist_corr_map']
        mean_dist_n_map = data_dict['accuracy']['mean_dist_clp_map']
        avg_var_post = data_dict['accuracy']['var_post']
        avg_var_map = data_dict['accuracy']['var_map']

        obs = data_dict['identifiers']['percentage_observed']
        additional_param = data_dict['identifiers']['additional_param']
        observed = data_dict['observed']

        # avg_var_all = avg_var.mean(dim=1).numpy()
        mean_dist_c_post_all = mean_dist_c_post.mean(dim=1).numpy()
        mean_dist_n_post_all = mean_dist_n_post.mean(dim=1).numpy()
        mean_dist_c_map_all = mean_dist_c_map.mean(dim=1).numpy()
        # mean_dist_c_map[observed] = float('nan')
        # mean_dist_c_map_all = mean_dist_c_map.nanmean(dim=1).numpy()
        mean_dist_n_map_all = mean_dist_n_map.mean(dim=1).numpy()
        avg_var_post_all = avg_var_post.mean(dim=1).numpy()
        avg_var_map_all = avg_var_map.mean(dim=1).numpy()

        if not add_param_available:
            # Create the statistics depending on whether the corresponding point was observed or not
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

            # Plots that do not sort the data according to the additional parameter and ignore it.
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
                'Femur reconstruction (LOOCV with N=10) using parallel MCMC sampling (10 chains with 10000 samples each)')
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
                'Femur reconstruction (LOOCV with N=10) using parallel MCMC sampling (10 chains with 10000 samples each)')
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
                'Femur reconstruction (LOOCV with N=10) using parallel MCMC sampling (10 chains, MAP sample selected for'
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
                'Femur reconstruction (LOOCV with N=10) using parallel MCMC sampling (10 chains, MAP sample selected for'
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
                'Femur reconstruction (LOOCV with N=10) using parallel MCMC sampling (10 chains with 10000 samples each)')
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
                'Femur reconstruction (LOOCV with N=10) using parallel MCMC sampling (10 chains, MAP sample selected for'
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
                'Femur reconstruction (LOOCV with N=10) using parallel MCMC sampling (10 chains with 10000 samples each)')
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
                'Femur reconstruction (LOOCV with N=10) using parallel MCMC sampling (10 chains with 10000 samples each)')
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
                'Femur reconstruction (LOOCV with N=10) using parallel MCMC sampling (10 chains, MAP sample selected for'
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
                'Femur reconstruction (LOOCV with N=47) using parallel MCMC sampling (10 chains, MAP sample selected for'
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
                'Femur reconstruction (LOOCV with N=10) using parallel MCMC sampling (10 chains with 10000 samples each)')
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
                'Femur reconstruction (LOOCV with N=10) using parallel MCMC sampling (10 chains, MAP sample selected for'
                'each one)')
            plt.ylim(0, 80)
            plt.tight_layout()
            filename = "map_var_all.png"
            png_plot_file = self.plot_dir / filename
            plt.savefig(png_plot_file)
            plt.close()

        else:
            # Plots that sort the data according to the additional parameter. This allows, for example, runs with a
            # different value of a specific hyperparameter to be compared.
            df = pd.DataFrame({
                'mean_dist_c_post_all': mean_dist_c_post_all,
                'mean_dist_n_post_all': mean_dist_n_post_all,
                'mean_dist_c_map_all': mean_dist_c_map_all,
                'mean_dist_n_map_all': mean_dist_n_map_all,
                'avg_var_post_all': avg_var_post_all,
                'avg_var_map_all': avg_var_map_all,
                'obs': obs.numpy(),
                'additional_param': additional_param.numpy() / 100
            })
            df_unique = df['obs'].unique()
            for group in df_unique:
                df_group = df[df['obs'] == group]

                # Average distance to corresponding point across all samples
                plt.figure(figsize=(20, 10))
                sns.boxplot(x='additional_param', y='mean_dist_c_post_all', data=df_group, color='skyblue')
                plt.title(f'Femur reconstruction (LOOCV with N=10) using parallel MCMC sampling (10 chains with 10000 '
                          f'samples each) with an observed portion of {group} per cent')
                plt.xlabel(r'Variance $\sigma^2$ used in the calculation of the likelihood term of the samples '
                           r'$[\mathrm{mm}^{2}]$')
                plt.ylabel(
                    r'$Average\ distance\ to\ the\ corresponding\ point\ on\ the\ reconstruction\ surface\ (utilising\ '
                    r'given\ correspondences)\ [\mathrm{mm}]$')
                plt.ylim(0, 25)
                plt.tight_layout()
                filename = f'post_dist_corr_{group}.png'
                png_plot_file = self.plot_dir / filename
                plt.savefig(png_plot_file)
                plt.close()

                # Average distance to the closest point across all samples
                plt.figure(figsize=(20, 10))
                sns.boxplot(x='additional_param', y='mean_dist_n_post_all', data=df_group, color='skyblue')
                plt.title(f'Femur reconstruction (LOOCV with N=10) using parallel MCMC sampling (10 chains with 10000 '
                          f'samples each) with an observed portion of {group} per cent')
                plt.xlabel(r'Variance $\sigma^2$ used in the calculation of the likelihood term of the samples '
                           r'$[\mathrm{mm}^{2}]$')
                plt.ylabel(r'$Average\ distance\ to\ the\ closest\ point\ on\ the\ reconstructed\ surface\ [\mathrm{mm}]$')
                plt.ylim(0, 25)
                plt.tight_layout()
                filename = f'post_dist_clp_{group}.png'
                png_plot_file = self.plot_dir / filename
                plt.savefig(png_plot_file)
                plt.close()

                # Average distance to the corresponding point across the MAP estimates of all MCMC chains
                plt.figure(figsize=(20, 10))
                sns.boxplot(x='additional_param', y='mean_dist_c_map_all', data=df_group, color='skyblue')
                plt.title(f'Femur reconstruction (LOOCV with N=10) using parallel MCMC sampling (10 chains, MAP sample '
                          f'selected for each one) with an observed portion of {group} per cent')
                plt.xlabel(r'Variance $\sigma^2$ used in the calculation of the likelihood term of the samples '
                           r'$[\mathrm{mm}^{2}]$')
                plt.ylabel(
                    r'$Average\ distance\ to\ the\ corresponding\ point\ on\ the\ reconstruction\ surface\ (utilising\ '
                    r'given\ correspondences)\ [\mathrm{mm}]$')
                plt.ylim(0, 25)
                plt.tight_layout()
                filename = f'map_dist_corr_{group}.png'
                png_plot_file = self.plot_dir / filename
                plt.savefig(png_plot_file)
                plt.close()

                # Average distance to the closest point across the MAP estimates of all MCMC chains
                plt.figure(figsize=(20, 10))
                sns.boxplot(x='additional_param', y='mean_dist_n_map_all', data=df_group, color='skyblue')
                plt.title(f'Femur reconstruction (LOOCV with N=10) using parallel MCMC sampling (10 chains, MAP sample '
                          f'selected for each one) with an observed portion of {group} per cent')
                plt.xlabel(r'Variance $\sigma^2$ used in the calculation of the likelihood term of the samples '
                           r'$[\mathrm{mm}^{2}]$')
                plt.ylabel(r'$Average\ distance\ to\ the\ closest\ point\ on\ the\ reconstructed\ surface\ [\mathrm{mm}]$')
                plt.ylim(0, 25)
                plt.tight_layout()
                filename = f'map_dist_clp_{group}.png'
                png_plot_file = self.plot_dir / filename
                plt.savefig(png_plot_file)
                plt.close()

                # Distances to corresponding points / variances across the MAP estimates of all MCMC chains combined
                fig, ax1 = plt.subplots(figsize=(20, 10))
                mean_of_means_dist = df_group.groupby('additional_param')['mean_dist_c_map_all'].mean().reset_index()
                ax1.scatter(mean_of_means_dist['additional_param'], mean_of_means_dist['mean_dist_c_map_all'], color='red',
                            label='Distances (left)', s=100)
                plt.plot(mean_of_means_dist['additional_param'], mean_of_means_dist['mean_dist_c_map_all'], color='red')
                ax1.set_xlabel(r'Variance $\sigma^2$ used in the calculation of the likelihood term of the samples '
                               r'$[\mathrm{mm}^{2}]$')
                ax1.set_ylabel(
                    r'$Average\ distance\ to\ the\ corresponding\ point\ on\ the\ reconstruction\ surface\ (utilising\ '
                    r'given\ correspondences)\ [\mathrm{mm}]$', color='red')
                ax1.set_ylim(0, 25)

                ax2 = ax1.twinx()
                mean_of_means_var = df_group.groupby('additional_param')['avg_var_post_all'].mean().reset_index()
                ax2.scatter(mean_of_means_var['additional_param'], mean_of_means_var['avg_var_post_all'], color='blue',
                            label='Variances (right)', s=100)
                plt.plot(mean_of_means_var['additional_param'], mean_of_means_var['avg_var_post_all'], color='blue')
                ax2.set_ylabel(r'$Average\ variance\ of\ the\ points\ [\mathrm{mm}^{2}]$', color='blue')
                ax2.set_ylim(0, 80)
                plt.title(f'Femur reconstruction (LOOCV with N=10) using parallel MCMC sampling (10 chains, MAP sample '
                          f'selected for each one) with an observed portion of {group} per cent')
                fig.tight_layout()
                filename = f'combined_map_dist_corr_var_{group}.png'
                png_plot_file = self.plot_dir / filename
                plt.savefig(png_plot_file)
                plt.close()





