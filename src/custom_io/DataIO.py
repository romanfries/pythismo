import pandas as pd
import json
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt
import plotly.io as pio
import plotly.graph_objects as go
import plotly.colors as pc

import numpy as np
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
        self.traceplot_dir = self.main_dir / Path('traceplots')
        self.samples_dir = self.main_dir / Path('samples')
        self.statistics_dir.mkdir(parents=True, exist_ok=True)
        self.chain_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.traceplot_dir.mkdir(parents=True, exist_ok=True)

    def rename_files(self):
        # TODO: Write the method properly by defining the new name(s) as an input parameter
        for file_path in self.statistics_dir.glob('mcmc_*_20_100.json'):
            parts = file_path.stem.split('_')
            xx = parts[1]

            new_filename = f'mcmc_{xx}_05_100.json'
            new_file_path = file_path.with_name(new_filename)
            file_path.rename(new_file_path)

    def write_statistics(self, data, loo, obs, additional_param):
        output_filename = f'mcmc_{loo}_{obs}_{additional_param}.json'
        output_file = self.statistics_dir / output_filename
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)

    def read_all_statistics(self):
        ess_list, rhat_list = [], []
        mean_dist_c_post_list, mean_dist_n_post_list, mean_dist_c_map_list, mean_dist_n_map_list = [], [], [], []
        squared_c_post_list, squared_n_post_list, squared_c_map_list, squared_n_map_list = [], [], [], []
        hausdorff_avg_list, hausdorff_map_list = [], []
        avg_var_list, avg_var_between_list = [], []
        means_list, mins_list, maxs_list, vars_list = [], [], [], []
        observed_list = []
        acc_par_list, acc_rnd_list, acc_trans_list, acc_rot_list, acc_tot_list = [], [], [], [], []
        # acc_par_list,  acc_trans_list, acc_rot_list, acc_tot_list = [], [], [], []
        loo_list, obs_list, additional_param_list = [], [], []

        json_files = sorted(self.statistics_dir.glob('*.json'))

        for json_file in json_files:
            with open(json_file, 'r') as f:
                loaded_data = json.load(f)

                ess = torch.tensor(loaded_data['effective_sample_sizes']['ess_per_param'])
                rhat = torch.tensor(loaded_data['effective_sample_sizes']['rhat_per_param'])

                ess_list.append(ess)
                rhat_list.append(rhat)

                mean_dist_c_post = torch.tensor(loaded_data['accuracy']['mean_dist_corr_post'])
                mean_dist_n_post = torch.tensor(loaded_data['accuracy']['mean_dist_clp_post'])
                squared_c_post = torch.tensor(loaded_data['accuracy']['squared_corr_post'])
                squared_n_post = torch.tensor(loaded_data['accuracy']['squared_clp_post'])
                mean_dist_c_map = torch.tensor(loaded_data['accuracy']['mean_dist_corr_map'])
                mean_dist_n_map = torch.tensor(loaded_data['accuracy']['mean_dist_clp_map'])
                squared_c_map = torch.tensor(loaded_data['accuracy']['squared_corr_map'])
                squared_n_map = torch.tensor(loaded_data['accuracy']['squared_clp_map'])
                hausdorff_avg = torch.tensor(loaded_data['accuracy']['hausdorff_avg'])
                hausdorff_map = loaded_data['accuracy']['hausdorff_map']
                avg_var = torch.tensor(loaded_data['accuracy']['var'])
                # avg_var_between = torch.tensor(loaded_data['accuracy']['var_between'])

                mean_dist_c_post_list.append(mean_dist_c_post)
                mean_dist_n_post_list.append(mean_dist_n_post)
                squared_c_post_list.append(squared_c_post)
                squared_n_post_list.append(squared_n_post)
                mean_dist_c_map_list.append(mean_dist_c_map)
                mean_dist_n_map_list.append(mean_dist_n_map)
                squared_c_map_list.append(squared_c_map)
                squared_n_map_list.append(squared_n_map)
                hausdorff_avg_list.append(hausdorff_avg)
                hausdorff_map_list.append(hausdorff_map)
                avg_var_list.append(avg_var)
                # avg_var_between_list.append(avg_var_between)

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
                acc_rnd = loaded_data['acceptance']['random_noise']
                acc_trans = loaded_data['acceptance']['translation']
                acc_rot = loaded_data['acceptance']['rotation']
                acc_tot = loaded_data['acceptance']['total']

                acc_par_list.append(acc_par)
                acc_rnd_list.append(acc_rnd)
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
            'effective_sample_sizes': torch.stack(ess_list),
            'accuracy': {
                'mean_dist_corr_post': pad_sequence([elem.T for elem in mean_dist_c_post_list],
                                                    batch_first=True, padding_value=float('nan')).permute(0, 2, 1),
                'mean_dist_clp_post': pad_sequence([elem.T for elem in mean_dist_n_post_list],
                                                   batch_first=True, padding_value=float('nan')).permute(0, 2, 1),
                'squared_corr_post': pad_sequence([elem.T for elem in squared_c_post_list], batch_first=True,
                                                  padding_value=float('nan')).permute(0, 2, 1),
                'squared_clp_post': pad_sequence([elem.T for elem in squared_n_post_list], batch_first=True,
                                                 padding_value=float('nan')).permute(0, 2, 1),
                'mean_dist_corr_map': torch.stack(mean_dist_c_map_list),
                'mean_dist_clp_map': torch.stack(mean_dist_n_map_list),
                'squared_corr_map': torch.stack(squared_c_map_list),
                'squared_clp_map': torch.stack(squared_n_map_list),
                'hausdorff_avg': pad_sequence([elem.T for elem in hausdorff_avg_list], batch_first=True,
                                              padding_value=float('nan')),
                'hausdorff_map': torch.tensor(hausdorff_map_list),
                'var': pad_sequence([elem.T for elem in avg_var_list], batch_first=True,
                                    padding_value=float('nan')).permute(0, 2, 1)
                # 'var_between': torch.stack(avg_var_between_list)
            },
            'unnormalised_log_density_posterior': {
                'mean': pad_sequence(means_list, batch_first=True, padding_value=float('nan')),
                'min': pad_sequence(mins_list, batch_first=True, padding_value=float('nan')),
                'max': pad_sequence(maxs_list, batch_first=True, padding_value=float('nan')),
                'var': pad_sequence(vars_list, batch_first=True, padding_value=float('nan'))
            },
            'observed': torch.stack(observed_list),
            'acceptance': {
                'model': torch.tensor(acc_par_list),
                'random_noise': torch.tensor(acc_rnd_list),
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
            rhat = torch.tensor(loaded_data['effective_sample_sizes']['rhat_per_param'])

            mean_dist_c_post = torch.tensor(loaded_data['accuracy']['mean_dist_corr_post'])
            mean_dist_n_post = torch.tensor(loaded_data['accuracy']['mean_dist_clp_post'])
            squared_c_post = torch.tensor(loaded_data['accuracy']['squared_corr_post'])
            squared_n_post = torch.tensor(loaded_data['accuracy']['squared_clp_post'])
            mean_dist_c_map = torch.tensor(loaded_data['accuracy']['mean_dist_corr_map'])
            mean_dist_n_map = torch.tensor(loaded_data['accuracy']['mean_dist_clp_map'])
            squared_c_map = torch.tensor(loaded_data['accuracy']['squared_corr_map'])
            squared_n_map = torch.tensor(loaded_data['accuracy']['squared_clp_map'])
            hausdorff_avg = torch.tensor(loaded_data['accuracy']['hausdorff_avg'])
            hausdorff_map = loaded_data['accuracy']['hausdorff_map']
            avg_var = torch.tensor(loaded_data['accuracy']['var'])
            # avg_var_between = torch.tensor(loaded_data['accuracy']['var_between'])

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
                'squared_corr_post': squared_c_post,
                'squared_clp_post': squared_n_post,
                'mean_dist_corr_map': mean_dist_c_map,
                'mean_dist_clp_map': mean_dist_n_map,
                'squared_corr_map': squared_c_map,
                'squared_clp_map': squared_n_map,
                'hausdorff_avg': hausdorff_avg,
                'hausdorff_map': hausdorff_map,
                'var': avg_var
                # 'var_between': avg_var_between
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

        colors = pc.sample_colorscale(pc.get_colorscale('Bluyl'), num_samples)
        for idx, mesh_ in enumerate(fig.data[:-1]):
            mesh_.opacity = 0.90
            mesh_.color = colors[idx]
        fig.data[-1].opacity = 0.90
        fig.data[-1].color = 'yellow'

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

    def save_target_map_dist(self, target, distances, loo, obs, additional_param, save_html=False):
        target.change_device('cpu')
        fig = go.Figure(data=[go.Mesh3d(x=target.tensor_points[:, 0].numpy(),
                                        y=target.tensor_points[:, 1].numpy(),
                                        z=target.tensor_points[:, 2].numpy(),
                                        i=target.cells[0].data[:, 0].numpy(),
                                        j=target.cells[0].data[:, 1].numpy(),
                                        k=target.cells[0].data[:, 2].numpy(),
                                        intensity=distances.cpu().numpy(),
                                        colorscale='bluered',
                                        cmin=0,
                                        cmax=25,
                                        colorbar=dict(title='Distance [mm]', thickness=20, x=0.7),
                                        intensitymode='vertex'
                                        )
                              ]
                        )
        fig.update_layout(
            title=f'Distance from MAP estimate to true target femur {loo} [mm]',
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='z'
            )
        )

        if save_html:
            output_filename = f'target_dist_{loo}_{obs}_{additional_param}.html'
        else:
            output_filename = f'target_dist_{loo}_{obs}_{additional_param}.png'
        output_file = self.samples_dir / output_filename

        if save_html:
            fig.write_html(output_file)
        else:
            fig.write_image(output_file, width=1920, height=1080)

    def save_target_avg(self, target, variances, loo, obs, additional_param, save_html=False):
        target.change_device('cpu')
        fig = go.Figure(data=[go.Mesh3d(x=target.tensor_points[:, 0].numpy(),
                                        y=target.tensor_points[:, 1].numpy(),
                                        z=target.tensor_points[:, 2].numpy(),
                                        i=target.cells[0].data[:, 0].numpy(),
                                        j=target.cells[0].data[:, 1].numpy(),
                                        k=target.cells[0].data[:, 2].numpy(),
                                        intensity=variances.cpu().numpy(),
                                        colorscale='bluered',
                                        cmin=0,
                                        cmax=130,
                                        colorbar=dict(title=r'Variance [$\mathrm{mm}^{2}$]', thickness=20, x=0.7),
                                        intensitymode='vertex'
                                        )
                              ]
                        )
        fig.update_layout(
            title=f'Average variance at each landmark for reconstructed target femur {loo}',
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='z'
            )
        )

        if save_html:
            output_filename = f'target_var_{loo}_{obs}_{additional_param}.html'
        else:
            output_filename = f'target_var_{loo}_{obs}_{additional_param}.png'
        output_file = self.samples_dir / output_filename

        if save_html:
            fig.write_html(output_file)
        else:
            fig.write_image(output_file, width=1920, height=1080)

    # TODO: Add an additional parameter for the following three read and write methods (see method above).
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

    def read_param_chain_posterior(self, loo, obs):
        input_filename = f'param_chain_{loo}_{obs}.pt'
        input_file = self.chain_dir / input_filename
        dict_param_chain_posterior = torch.load(input_file)
        return dict_param_chain_posterior

    def write_traceplots(self, traceplots, loo, obs, additional_param):
        dir = self.traceplot_dir / Path(f'traceplots_{loo}_{obs}_{additional_param}')
        dir.mkdir(parents=True, exist_ok=True)
        for chain, traceplot in enumerate(traceplots):
            output_filename = f'log_density_posterior_{chain}.png'
            output_file = dir / output_filename
            pio.write_image(traceplot, output_file)

    def generate_plots(self, num_chains_select, data_dict=None, add_param_available=False):
        # Needs to be adjusted depending on the experiments performed
        if data_dict is None:
            data_dict = self.read_all_statistics()

        # Prepare the data
        mean_dist_c_post = data_dict['accuracy']['mean_dist_corr_post']
        mean_dist_n_post = data_dict['accuracy']['mean_dist_clp_post']
        squared_c_post = data_dict['accuracy']['squared_corr_post']
        # squared_c_post = torch.ones_like(mean_dist_c_post)
        squared_n_post = data_dict['accuracy']['squared_clp_post']
        # squared_n_post = torch.ones_like(mean_dist_n_post)
        mean_dist_c_map = data_dict['accuracy']['mean_dist_corr_map']
        mean_dist_n_map = data_dict['accuracy']['mean_dist_clp_map']
        squared_c_map = data_dict['accuracy']['squared_corr_map']
        # squared_c_map = torch.ones_like(mean_dist_c_map)
        squared_n_map = data_dict['accuracy']['squared_clp_map']
        # squared_n_map = torch.ones_like(mean_dist_n_map)
        hausdorff_avg = data_dict['accuracy']['hausdorff_avg']
        hausdorff_map = data_dict['accuracy']['hausdorff_map']
        avg_var = data_dict['accuracy']['var']
        # avg_var_between = data_dict['accuracy']['var_between']

        shape = data_dict['identifiers']['reconstructed_shape']
        obs = data_dict['identifiers']['percentage_observed']
        additional_param = data_dict['identifiers']['additional_param']
        observed = data_dict['observed']

        # For every femur shape, keep only 'num_chains' randomly selected chains
        num_shapes, num_points, max_chains = mean_dist_c_post.size()
        nan_mask = torch.isnan(mean_dist_c_post).any(dim=1)
        permutation_int = torch.stack([torch.randperm(max_chains) for _ in range(num_shapes)])
        permutation = permutation_int.float()
        permutation[nan_mask[torch.arange(num_shapes).unsqueeze(1), permutation_int]] = float('nan')
        new_mask = ~torch.isnan(permutation)
        permutation = permutation.gather(1,
                                         torch.argsort(new_mask, dim=1, descending=True)[:, :num_chains_select]).long()

        # avg_var_all = avg_var.mean(dim=1).numpy()
        mean_dist_c_post = torch.gather(mean_dist_c_post, dim=2, index=permutation.unsqueeze(1).expand(-1, 200, -1))
        mean_dist_n_post = torch.gather(mean_dist_n_post, dim=2, index=permutation.unsqueeze(1).expand(-1, 200, -1))
        squared_c_post = torch.gather(squared_c_post, dim=2, index=permutation.unsqueeze(1).expand(-1, 200, -1))
        squared_n_post = torch.gather(squared_n_post, dim=2, index=permutation.unsqueeze(1).expand(-1, 200, -1))
        hausdorff_avg = torch.gather(hausdorff_avg, dim=1, index=permutation)
        avg_var = torch.gather(avg_var, dim=2, index=permutation.unsqueeze(1).expand(-1, 200, -1))

        # Average out all the points and chains
        mean_dist_c_post_all = mean_dist_c_post.nanmean(dim=2).nanmean(dim=1).numpy()
        mean_dist_n_post_all = mean_dist_n_post.nanmean(dim=2).nanmean(dim=1).numpy()
        squared_c_post_all = squared_c_post.nanmean(dim=2).nanmean(dim=1).numpy()
        squared_n_post_all = squared_n_post.nanmean(dim=2).nanmean(dim=1).numpy()
        mean_dist_c_map_all = mean_dist_c_map.nanmean(dim=1).numpy()
        # mean_dist_c_map[observed] = float('nan')
        # mean_dist_c_map_all = mean_dist_c_map.nanmean(dim=1).numpy()
        mean_dist_n_map_all = mean_dist_n_map.nanmean(dim=1).numpy()
        squared_c_map_all = squared_c_map.nanmean(dim=1).numpy()
        squared_n_map_all = squared_n_map.nanmean(dim=1).numpy()
        hausdorff_avg_all = hausdorff_avg.nanmean(dim=1).numpy()
        hausdorff_map = hausdorff_map.numpy()
        avg_var_all = avg_var.nanmean(dim=2).nanmean(dim=1).numpy()
        # avg_var_between_all = avg_var_between.nanmean(dim=1).numpy()

        # Create the statistics depending on whether the corresponding point was observed or not
        # avg_var_copy_t, avg_var_copy_f = avg_var.clone(), avg_var.clone()
        mean_dist_c_post_copy_t, mean_dist_c_post_copy_f = mean_dist_c_post.clone(), mean_dist_c_post.clone()
        mean_dist_n_post_copy_t, mean_dist_n_post_copy_f = mean_dist_n_post.clone(), mean_dist_n_post.clone()
        squared_c_post_copy_t, squared_c_post_copy_f = squared_c_post.clone(), squared_c_post.clone()
        squared_n_post_copy_t, squared_n_post_copy_f = squared_n_post.clone(), squared_n_post.clone()
        mean_dist_c_map_copy_t, mean_dist_c_map_copy_f = mean_dist_c_map.clone(), mean_dist_c_map.clone()
        mean_dist_n_map_copy_t, mean_dist_n_map_copy_f = mean_dist_n_map.clone(), mean_dist_n_map.clone()
        squared_c_map_copy_t, squared_c_map_copy_f = squared_c_map.clone(), squared_c_map.clone()
        squared_n_map_copy_t, squared_n_map_copy_f = squared_n_map.clone(), squared_n_map.clone()
        avg_var_copy_t, avg_var_copy_f = avg_var.clone(), avg_var.clone()
        # avg_var_between_copy_t, avg_var_between_copy_f = avg_var_between.clone(), avg_var_between.clone()

        # avg_var_copy_t[~obs] = float('nan')
        mean_dist_c_post_copy_t[~observed] = float('nan')
        mean_dist_n_post_copy_t[~observed] = float('nan')
        squared_c_post_copy_t[~observed] = float('nan')
        squared_n_post_copy_t[~observed] = float('nan')
        mean_dist_c_map_copy_t[~observed] = float('nan')
        mean_dist_n_map_copy_t[~observed] = float('nan')
        squared_c_map_copy_t[~observed] = float('nan')
        squared_n_map_copy_t[~observed] = float('nan')
        avg_var_copy_t[~observed] = float('nan')
        # avg_var_between_copy_t[~observed] = float('nan')

        # avg_var_mean_t = torch.nanmean(avg_var_copy_t, dim=1)
        mean_dist_c_post_mean_t = torch.nanmean(torch.nanmean(mean_dist_c_post_copy_t, dim=2), dim=1)
        mean_dist_n_post_mean_t = torch.nanmean(torch.nanmean(mean_dist_n_post_copy_t, dim=2), dim=1)
        squared_c_post_mean_t = torch.nanmean(torch.nanmean(squared_c_post_copy_t, dim=2), dim=1)
        squared_n_post_mean_t = torch.nanmean(torch.nanmean(squared_n_post_copy_t, dim=2), dim=1)
        mean_dist_c_map_mean_t = torch.nanmean(mean_dist_c_map_copy_t, dim=1)
        mean_dist_n_map_mean_t = torch.nanmean(mean_dist_n_map_copy_t, dim=1)
        squared_c_map_mean_t = torch.nanmean(squared_c_map_copy_t, dim=1)
        squared_n_map_mean_t = torch.nanmean(squared_n_map_copy_t, dim=1)
        avg_var_mean_t = torch.nanmean(torch.nanmean(avg_var_copy_t, dim=2), dim=1)
        # avg_var_between_mean_t = torch.nanmean(avg_var_between_copy_t, dim=1)

        # avg_var_copy_f[obs] = float('nan')
        mean_dist_c_post_copy_f[observed] = float('nan')
        mean_dist_n_post_copy_f[observed] = float('nan')
        squared_c_post_copy_f[observed] = float('nan')
        squared_n_post_copy_f[observed] = float('nan')
        mean_dist_c_map_copy_f[observed] = float('nan')
        mean_dist_n_map_copy_f[observed] = float('nan')
        squared_c_map_copy_f[observed] = float('nan')
        squared_n_map_copy_f[observed] = float('nan')
        avg_var_copy_f[observed] = float('nan')
        # avg_var_between_copy_f[observed] = float('nan')

        # avg_var_mean_f = torch.nanmean(avg_var_copy_f, dim=1)
        mean_dist_c_post_mean_f = torch.nanmean(torch.nanmean(mean_dist_c_post_copy_f, dim=2), dim=1)
        mean_dist_n_post_mean_f = torch.nanmean(torch.nanmean(mean_dist_n_post_copy_f, dim=2), dim=1)
        squared_c_post_mean_f = torch.nanmean(torch.nanmean(squared_c_post_copy_f, dim=2), dim=1)
        squared_n_post_mean_f = torch.nanmean(torch.nanmean(squared_n_post_copy_f, dim=2), dim=1)
        mean_dist_c_map_mean_f = torch.nanmean(mean_dist_c_map_copy_f, dim=1)
        mean_dist_n_map_mean_f = torch.nanmean(mean_dist_n_map_copy_f, dim=1)
        squared_c_map_mean_f = torch.nanmean(squared_c_map_copy_f, dim=1)
        squared_n_map_mean_f = torch.nanmean(squared_n_map_copy_f, dim=1)
        avg_var_mean_f = torch.nanmean(torch.nanmean(avg_var_copy_f, dim=2), dim=1)
        # avg_var_between_mean_f = torch.nanmean(avg_var_between_copy_f, dim=1)

        df = pd.DataFrame({
            'mean_dist_c_post_all': mean_dist_c_post_all,
            'mean_dist_c_post_mean_t': mean_dist_c_post_mean_t,
            'mean_dist_c_post_mean_f': mean_dist_c_post_mean_f,
            'mean_dist_n_post_all': mean_dist_n_post_all,
            'mean_dist_n_post_mean_t': mean_dist_n_post_mean_t,
            'mean_dist_n_post_mean_f': mean_dist_n_post_mean_f,
            'squared_c_post_all': squared_c_post_all,
            'squared_c_post_mean_t': squared_c_post_mean_t,
            'squared_c_post_mean_f': squared_c_post_mean_f,
            'squared_n_post_all': squared_n_post_all,
            'squared_n_post_mean_t': squared_n_post_mean_t,
            'squared_n_post_mean_f': squared_n_post_mean_f,
            'mean_dist_c_map_all': mean_dist_c_map_all,
            'mean_dist_c_map_mean_t': mean_dist_c_map_mean_t,
            'mean_dist_c_map_mean_f': mean_dist_c_map_mean_f,
            'mean_dist_n_map_all': mean_dist_n_map_all,
            'mean_dist_n_map_mean_t': mean_dist_n_map_mean_t,
            'mean_dist_n_map_mean_f': mean_dist_n_map_mean_f,
            'squared_c_map_all': squared_c_map_all,
            'squared_c_map_mean_t': squared_c_map_mean_t,
            'squared_c_map_mean_f': squared_c_map_mean_f,
            'squared_n_map_all': squared_n_map_all,
            'squared_n_map_mean_t': squared_n_map_mean_t,
            'squared_n_map_mean_f': squared_n_map_mean_f,
            'hausdorff_avg': hausdorff_avg_all,
            'hausdorff_map': hausdorff_map,
            'avg_var_all': avg_var_all,
            'avg_var_mean_t': avg_var_mean_t,
            'avg_var_mean_f': avg_var_mean_f,
            # 'avg_var_between_all': avg_var_between_all,
            # 'avg_var_between_mean_t': avg_var_between_mean_t,
            # 'avg_var_between_mean_f': avg_var_between_mean_f,
            'shape': shape.numpy(),
            'obs': obs.numpy(),
            'additional_param': additional_param.numpy() / 100
        })

        if not add_param_available:
            # Plots that do not sort the data according to the additional parameter and ignore it.
            # Average distance to corresponding point across all samples
            df_melted = df.melt(id_vars=['obs'],
                                value_vars=['mean_dist_c_post_all', 'mean_dist_c_post_mean_t',
                                            'mean_dist_c_post_mean_f'],
                                var_name='category', value_name='mean')

            category_mapping = {
                'mean_dist_c_post_all': 'All points',
                'mean_dist_c_post_mean_t': 'Observed points',
                'mean_dist_c_post_mean_f': 'Reconstructed points'
            }

            obs_mapping = {
                10: 50,
                20: 100,
                30: 125,
                40: 150,
                50: 175,
                60: 200
            }

            df_melted['category'] = df_melted['category'].map(category_mapping)
            df_melted['obs'] = df_melted['obs'].map(obs_mapping)
            df_melted = df_melted.dropna(subset=['mean'])

            plt.figure(figsize=(20, 10), dpi=300)

            sns.boxplot(x='obs', y='mean', hue='category', data=df_melted)

            plt.xlabel(r'Weighting factor $\gamma$ of the likelihood term')
            plt.ylabel(
                r'Average distance to the corresponding point on the reconstruction surface (utilising '
                r'given correspondences) $[\mathrm{mm}]$')
            plt.title(
                'Femur reconstruction (LOOCV with N=47) using parallel MCMC sampling (30 chains with 40000 samples each) with target-aware model')
            plt.legend(title='Type of points analysed', bbox_to_anchor=(1.05, 1), loc='upper right')
            plt.ylim(0, 40)
            # plt.xticks(custom_ticks, custom_labels)
            plt.tight_layout()
            filename = "post_dist_split.png"
            png_plot_file = self.plot_dir / filename
            plt.savefig(png_plot_file)
            plt.close()

            # Average distance to corresponding point across all samples
            df_small = pd.DataFrame({
                'obs': obs.numpy(),
                'mean_dist_c_post_all': mean_dist_c_post_all
            })

            obs_mapping = {
                10: 50,
                20: 100,
                30: 125,
                40: 150,
                50: 175,
                60: 200
            }
            df_small['obs'] = df_small['obs'].map(obs_mapping)

            plt.figure(figsize=(20, 10), dpi=300)
            sns.boxplot(x='obs', y='mean_dist_c_post_all', data=df_small, color='skyblue')

            plt.xlabel(r'Weighting factor $\gamma$ of the likelihood term')
            plt.ylabel(
                r'Average distance to the corresponding point on the reconstruction surface (utilising '
                r'given correspondences) $[\mathrm{mm}]$')
            plt.title(
                'Femur reconstruction (LOOCV with N=47) using parallel MCMC sampling (30 chains with 40000 samples each) with target-aware model')
            plt.ylim(0, 25)
            plt.tight_layout()
            filename = "post_dist.png"
            png_plot_file = self.plot_dir / filename
            plt.savefig(png_plot_file)
            plt.close()

            # Average variance of the points across all chains and samples
            df_melted = df.melt(id_vars=['obs'],
                                value_vars=['avg_var_all', 'avg_var_mean_t', 'avg_var_mean_f'],
                                var_name='category', value_name='mean')

            category_mapping = {
                'avg_var_all': 'All points',
                'avg_var_mean_t': 'Observed points',
                'avg_var_mean_f': 'Reconstructed points'
            }

            obs_mapping = {
                10: 50,
                20: 100,
                30: 125,
                40: 150,
                50: 175,
                60: 200
            }

            df_melted['category'] = df_melted['category'].map(category_mapping)
            df_melted['obs'] = df_melted['obs'].map(obs_mapping)
            df_melted = df_melted.dropna(subset=['mean'])

            plt.figure(figsize=(20, 10), dpi=300)

            sns.boxplot(x='obs', y='mean', hue='category', data=df_melted)

            plt.xlabel(r'Weighting factor $\gamma$ of the likelihood term')
            plt.ylabel(
                r'Average variance of the points $[\mathrm{mm}^{2}]$')
            plt.title(
                'Femur reconstruction (LOOCV with N=47) using parallel MCMC sampling (30 chains with 40000 samples each) with target-aware model')
            plt.legend(title='Type of points analysed', bbox_to_anchor=(1.05, 1), loc='upper right')
            plt.ylim(0, 150)
            plt.tight_layout()
            filename = "var_split.png"
            png_plot_file = self.plot_dir / filename
            plt.savefig(png_plot_file)
            plt.close()

            # Average distance to the corresponding point across the MAP estimates of all MCMC chains
            df_melted = df.melt(id_vars=['obs'],
                                value_vars=['mean_dist_c_map_all', 'mean_dist_c_map_mean_t',
                                            'mean_dist_c_map_mean_f'],
                                var_name='category', value_name='mean')

            category_mapping = {
                'mean_dist_c_map_all': 'All points',
                'mean_dist_c_map_mean_t': 'Observed points',
                'mean_dist_c_map_mean_f': 'Reconstructed points'
            }

            obs_mapping = {
                10: 50,
                20: 100,
                30: 125,
                40: 150,
                50: 175,
                60: 200
            }

            df_melted['category'] = df_melted['category'].map(category_mapping)
            df_melted['obs'] = df_melted['obs'].map(obs_mapping)
            df_melted = df_melted.dropna(subset=['mean'])

            plt.figure(figsize=(20, 10), dpi=300)

            sns.boxplot(x='obs', y='mean', hue='category', data=df_melted)

            plt.xlabel(r'Weighting factor $\gamma$ of the likelihood term')
            plt.ylabel(
                r'Average distance to the corresponding point on the reconstruction surface (utilising '
                r'given correspondences) $[\mathrm{mm}]$')
            plt.title(
                'Femur reconstruction (LOOCV with N=47) using parallel MCMC sampling (30 chains, MAP sample selected) with target-aware model')
            plt.legend(title='Type of points analysed', bbox_to_anchor=(1.05, 1), loc='upper right')
            plt.ylim(0, 40)
            # plt.xticks(custom_ticks, custom_labels)
            plt.tight_layout()
            filename = "map_dist_split.png"
            png_plot_file = self.plot_dir / filename
            plt.savefig(png_plot_file)
            plt.close()

            # Average distance to the corresponding point across the MAP estimates of all MCMC chains
            df_small = pd.DataFrame({
                'obs': obs.numpy(),
                'mean_dist_c_map_all': mean_dist_c_map_all
            })

            obs_mapping = {
                10: 50,
                20: 100,
                30: 125,
                40: 150,
                50: 175,
                60: 200
            }
            df_small['obs'] = df_small['obs'].map(obs_mapping)

            plt.figure(figsize=(20, 10), dpi=300)
            sns.boxplot(x='obs', y='mean_dist_c_map_all', data=df_small, color='skyblue')

            plt.xlabel(r'Weighting factor $\gamma$ of the likelihood term')
            plt.ylabel(
                r'Average distance to the corresponding point on the reconstruction surface (utilising '
                r'given correspondences) $[\mathrm{mm}]$')
            plt.title(
                'Femur reconstruction (LOOCV with N=47) using parallel MCMC sampling (30 chains, MAP sample selected) with target-aware model')
            plt.ylim(0, 25)
            plt.tight_layout()
            filename = "map_dist.png"
            png_plot_file = self.plot_dir / filename
            plt.savefig(png_plot_file)
            plt.close()

            # Average squared distance to corresponding point across all samples
            df_melted = df.melt(id_vars=['obs'],
                                value_vars=['squared_c_post_all', 'squared_c_post_mean_t',
                                            'squared_c_post_mean_f'],
                                var_name='category', value_name='mean')

            category_mapping = {
                'squared_c_post_all': 'All points',
                'squared_c_post_mean_t': 'Observed points',
                'squared_c_post_mean_f': 'Reconstructed points'
            }

            obs_mapping = {
                10: 50,
                20: 100,
                30: 125,
                40: 150,
                50: 175,
                60: 200
            }

            df_melted['category'] = df_melted['category'].map(category_mapping)
            df_melted['obs'] = df_melted['obs'].map(obs_mapping)
            df_melted = df_melted.dropna(subset=['mean'])

            plt.figure(figsize=(20, 10), dpi=300)

            sns.boxplot(x='obs', y='mean', hue='category', data=df_melted)

            plt.xlabel(r'Weighting factor $\gamma$ of the likelihood term')
            plt.ylabel(
                r'Average squared distance to the corresponding point on the reconstruction surface (utilising '
                r'given correspondences) $[\mathrm{mm}^{2}]$')
            plt.title(
                'Femur reconstruction (LOOCV with N=47) using parallel MCMC sampling (30 chains with 40000 samples each) with target-aware model')
            plt.legend(title='Type of points analysed', bbox_to_anchor=(1.05, 1), loc='upper right')
            plt.ylim(0, 1600)
            # plt.xticks(custom_ticks, custom_labels)
            plt.tight_layout()
            filename = "post_squared_split.png"
            png_plot_file = self.plot_dir / filename
            plt.savefig(png_plot_file)
            plt.close()

            # Average squared distance to corresponding point across all samples
            df_small = pd.DataFrame({
                'obs': obs.numpy(),
                'squared_c_post_all': squared_c_post_all
            })

            obs_mapping = {
                10: 50,
                20: 100,
                30: 125,
                40: 150,
                50: 175,
                60: 200
            }
            df_small['obs'] = df_small['obs'].map(obs_mapping)

            plt.figure(figsize=(20, 10), dpi=300)
            sns.boxplot(x='obs', y='squared_c_post_all', data=df_small, color='skyblue')

            plt.xlabel(r'Weighting factor $\gamma$ of the likelihood term')
            plt.ylabel(
                r'Average squared distance to the corresponding point on the reconstruction surface (utilising '
                r'given correspondences) $[\mathrm{mm}^{2}]$')
            plt.title(
                'Femur reconstruction (LOOCV with N=47) using parallel MCMC sampling (30 chains with 40000 samples each) with target-aware model')
            plt.ylim(0, 800)
            plt.tight_layout()
            filename = "post_squared.png"
            png_plot_file = self.plot_dir / filename
            plt.savefig(png_plot_file)
            plt.close()

            # Average squared distance to the corresponding point across the MAP estimates of all MCMC chains
            df_melted = df.melt(id_vars=['obs'],
                                value_vars=['squared_c_map_all', 'squared_c_map_mean_t',
                                            'squared_c_map_mean_f'],
                                var_name='category', value_name='mean')

            category_mapping = {
                'squared_c_map_all': 'All points',
                'squared_c_map_mean_t': 'Observed points',
                'squared_c_map_mean_f': 'Reconstructed points'
            }

            obs_mapping = {
                10: 50,
                20: 100,
                30: 125,
                40: 150,
                50: 175,
                60: 200
            }

            df_melted['category'] = df_melted['category'].map(category_mapping)
            df_melted['obs'] = df_melted['obs'].map(obs_mapping)
            df_melted = df_melted.dropna(subset=['mean'])

            plt.figure(figsize=(20, 10), dpi=300)

            sns.boxplot(x='obs', y='mean', hue='category', data=df_melted)

            plt.xlabel(r'Weighting factor $\gamma$ of the likelihood term')
            plt.ylabel(
                r'Average squared distance to the corresponding point on the reconstruction surface (utilising '
                r'given correspondences) $[\mathrm{mm}^{2}]$')
            plt.title(
                'Femur reconstruction (LOOCV with N=47) using parallel MCMC sampling (30 chains, MAP sample selected) with target-aware model')
            plt.legend(title='Type of points analysed', bbox_to_anchor=(1.05, 1), loc='upper right')
            plt.ylim(0, 1600)
            # plt.xticks(custom_ticks, custom_labels)
            plt.tight_layout()
            filename = "map_squared_split.png"
            png_plot_file = self.plot_dir / filename
            plt.savefig(png_plot_file)
            plt.close()

            # Average distance to the corresponding point across the MAP estimates of all MCMC chains
            df_small = pd.DataFrame({
                'obs': obs.numpy(),
                'squared_c_map_all': squared_c_map_all
            })

            obs_mapping = {
                10: 50,
                20: 100,
                30: 125,
                40: 150,
                50: 175,
                60: 200
            }
            df_small['obs'] = df_small['obs'].map(obs_mapping)

            plt.figure(figsize=(20, 10), dpi=300)
            sns.boxplot(x='obs', y='squared_c_map_all', data=df_small, color='skyblue')

            plt.xlabel(r'Weighting factor $\gamma$ of the likelihood term')
            plt.ylabel(
                r'Average squared distance to the corresponding point on the reconstruction surface (utilising '
                r'given correspondences) $[\mathrm{mm}^{2}]$')
            plt.title(
                'Femur reconstruction (LOOCV with N=47) using parallel MCMC sampling (30 chains, MAP sample selected) with target-aware model')
            plt.ylim(0, 800)
            plt.tight_layout()
            filename = "map_squared.png"
            png_plot_file = self.plot_dir / filename
            plt.savefig(png_plot_file)
            plt.close()

            # Average Hausdorff distance between the MAP estimates and their corresponding true shape
            df_small = pd.DataFrame({
                'obs': obs.numpy(),
                'hausdorff_map': hausdorff_map
            })

            obs_mapping = {
                10: 50,
                20: 100,
                30: 125,
                40: 150,
                50: 175,
                60: 200
            }
            df_small['obs'] = df_small['obs'].map(obs_mapping)

            plt.figure(figsize=(20, 10), dpi=300)
            sns.boxplot(x='obs', y='hausdorff_map', data=df_small, color='skyblue')

            plt.xlabel(r'Weighting factor $\gamma$ of the likelihood term')
            plt.ylabel(
                r'Average Hausdorff distances between MAP estimate and true target $[\mathrm{mm}]$')
            plt.title(
                'Femur reconstruction (LOOCV with N=47) using parallel MCMC sampling (30 chains, MAP sample selected) with target-aware model')
            plt.ylim(0, 80)
            plt.tight_layout()
            filename = "map_hausdorff.png"
            png_plot_file = self.plot_dir / filename
            plt.savefig(png_plot_file)
            plt.close()

            # Average Hausdorff distance between all samples and their corresponding true shape
            df_small = pd.DataFrame({
                'obs': obs.numpy(),
                'hausdorff_post': hausdorff_avg_all
            })

            obs_mapping = {
                10: 50,
                20: 100,
                30: 125,
                40: 150,
                50: 175,
                60: 200
            }
            df_small['obs'] = df_small['obs'].map(obs_mapping)

            plt.figure(figsize=(20, 10), dpi=300)
            sns.boxplot(x='obs', y='hausdorff_post', data=df_small, color='skyblue')

            plt.xlabel(r'Weighting factor $\gamma$ of the likelihood term')
            plt.ylabel(
                r'Average Hausdorff distances between all samples and their corresponding true target $[\mathrm{mm}]$')
            plt.title(
                'Femur reconstruction (LOOCV with N=47) using parallel MCMC sampling (30 chains with 40000 samples each) with target-aware model')
            plt.ylim(0, 80)
            plt.tight_layout()
            filename = "post_hausdorff.png"
            png_plot_file = self.plot_dir / filename
            plt.savefig(png_plot_file)
            plt.close()

            # Split by reconstructed shape
            # Average distance to corresponding point across all samples
            unique_shapes = np.sort(df['shape'].unique())
            for i in range(0, len(unique_shapes), 5):
                subset = unique_shapes[i:i+5]
                df_subset = df[df['shape'].isin(subset)]

                plt.figure(figsize=(20, 10), dpi=300)
                pos_start = 0

                for shape in subset:
                    shape_data = df_subset[df_subset['shape'] == shape]
                    grouped = shape_data.groupby('obs')['mean_dist_c_post_all'].mean()
                    plt.bar(np.arange(len(grouped)) + pos_start, grouped.values, width=0.8, label=f'{shape}')
                    pos_start += len(grouped) + 2

                plt.title(
                    'Femur reconstruction errors using parallel MCMC sampling (30 chains with 40000 samples each) '
                    'split by reconstructed shape')
                plt.ylim(0, 40)
                plt.xlabel(r'Weighting factor $\gamma$ of the likelihood term')
                plt.ylabel(r'Average distance to the corresponding point on the reconstruction surface (utilising'
                            'given correspondences) $[\mathrm{mm}]$')
                plt.legend(title='Reconstructed Femur Bone')

                if i == 45:
                    x_labels = [50, 100, 125, 150, 175, 200] * 2
                else:
                    x_labels = [50, 100, 125, 150, 175, 200] * 5
                x_tick_positions = []
                pos_x = 0
                for shape in subset:
                    shape_data = df_subset[df_subset['shape'] == shape]
                    grouped = shape_data.groupby('obs')['mean_dist_c_post_all'].mean()
                    for _ in grouped.index:
                        x_tick_positions.append(pos_x)
                        pos_x += 1
                    pos_x += 2

                plt.xticks(ticks=x_tick_positions, labels=x_labels, rotation=90)
                plt.tight_layout()

                filename = f'post_dist_per_shape_{i}.png'
                split_dir = self.plot_dir / Path('split_by_shape')
                split_dir.mkdir(parents=True, exist_ok=True)
                png_plot_file = split_dir / filename
                plt.savefig(png_plot_file)
                plt.close()

            # Average distance to the corresponding point across the MAP estimates of all MCMC chains
            for i in range(0, len(unique_shapes), 5):
                subset = unique_shapes[i:i+5]
                df_subset = df[df['shape'].isin(subset)]

                plt.figure(figsize=(20, 10), dpi=300)
                pos_start = 0

                for shape in subset:
                    shape_data = df_subset[df_subset['shape'] == shape]
                    grouped = shape_data.groupby('obs')['mean_dist_c_map_all'].mean()
                    plt.bar(np.arange(len(grouped)) + pos_start, grouped.values, width=0.8, label=f'{shape}')
                    pos_start += len(grouped) + 2

                plt.title(
                    'Femur reconstruction errors using parallel MCMC sampling (MAP sample selected) split by '
                    'reconstructed shape')
                plt.ylim(0, 40)
                plt.xlabel(r'Weighting factor $\gamma$ of the likelihood term')
                plt.ylabel(r'Average distance to the corresponding point on the reconstruction surface (utilising'
                            'given correspondences) $[\mathrm{mm}]$')
                plt.legend(title='Reconstructed Femur Bone')

                if i == 45:
                    x_labels = [50, 100, 125, 150, 175, 200] * 2
                else:
                    x_labels = [50, 100, 125, 150, 175, 200] * 5
                x_tick_positions = []
                pos_x = 0
                for shape in subset:
                    shape_data = df_subset[df_subset['shape'] == shape]
                    grouped = shape_data.groupby('obs')['mean_dist_c_map_all'].mean()
                    for _ in grouped.index:
                        x_tick_positions.append(pos_x)
                        pos_x += 1
                    pos_x += 2

                plt.xticks(ticks=x_tick_positions, labels=x_labels, rotation=90)
                plt.tight_layout()

                filename = f'map_dist_per_shape_{i}.png'
                split_dir = self.plot_dir / Path('split_by_shape')
                split_dir.mkdir(parents=True, exist_ok=True)
                png_plot_file = split_dir / filename
                plt.savefig(png_plot_file)
                plt.close()

        else:
            # Plots that sort the data according to the additional parameter. This allows, for example, runs with a
            # different value of a specific hyperparameter to be compared.
            df_unique = df['obs'].unique()
            # The same plots are generated for each different observed proportion
            for group in df_unique:
                df_group = df[df['obs'] == group]

                # Average distance to corresponding point across all samples
                plt.figure(figsize=(20, 10), dpi=300)
                sns.boxplot(x='additional_param', y='mean_dist_c_post_all', data=df_group, color='skyblue')
                plt.title(f'Femur reconstruction (LOOCV with N=10) using parallel MCMC sampling (50 chains with 50000 '
                          f'samples each) with an observed portion of {group} per cent')
                plt.xlabel(r'Variance $\sigma^2$ used in the calculation of the likelihood term of the samples '
                           r'$[\mathrm{mm}^{2}]$')
                plt.ylabel(
                    r'$Average\ distance\ to\ the\ corresponding\ point\ on\ the\ reconstruction\ surface\ (utilising\ '
                    r'given\ fixed_correspondences)\ [\mathrm{mm}]$')
                plt.ylim(0, 25)
                plt.tight_layout()
                filename = f'post_dist_corr_{group}.png'
                png_plot_file = self.plot_dir / filename
                plt.savefig(png_plot_file)
                plt.close()

                # Average distance to the closest point across all samples
                plt.figure(figsize=(20, 10))
                sns.boxplot(x='additional_param', y='mean_dist_n_post_all', data=df_group, color='skyblue')
                plt.title(f'Femur reconstruction (LOOCV with N=10) using parallel MCMC sampling (50 chains with 50000 '
                          f'samples each) with an observed portion of {group} per cent')
                plt.xlabel(r'Variance $\sigma^2$ used in the calculation of the likelihood term of the samples '
                           r'$[\mathrm{mm}^{2}]$')
                plt.ylabel(
                    r'$Average\ distance\ to\ the\ closest\ point\ on\ the\ reconstructed\ surface\ [\mathrm{mm}]$')
                plt.ylim(0, 25)
                plt.tight_layout()
                filename = f'post_dist_clp_{group}.png'
                png_plot_file = self.plot_dir / filename
                plt.savefig(png_plot_file)
                plt.close()

                # Average distance to the corresponding point across the MAP estimate of all MCMC runs
                plt.figure(figsize=(20, 10))
                sns.boxplot(x='additional_param', y='mean_dist_c_map_all', data=df_group, color='skyblue')
                plt.title(f'Femur reconstruction (LOOCV with N=10) using parallel MCMC sampling (50 chains, MAP sample '
                          f'selected) with an observed portion of {group} per cent')
                plt.xlabel(r'Variance $\sigma^2$ used in the calculation of the likelihood term of the samples '
                           r'$[\mathrm{mm}^{2}]$')
                plt.ylabel(
                    r'$Average\ distance\ to\ the\ corresponding\ point\ on\ the\ reconstruction\ surface\ (utilising\ '
                    r'given\ fixed_correspondences)\ [\mathrm{mm}]$')
                plt.ylim(0, 25)
                plt.tight_layout()
                filename = f'map_dist_corr_{group}.png'
                png_plot_file = self.plot_dir / filename
                plt.savefig(png_plot_file)
                plt.close()

                # Same plot as above but distinguish whether points were observed or not
                df_melted = df.melt(id_vars=['additional_param'],
                                    value_vars=['mean_dist_c_map_all', 'mean_dist_c_map_mean_t',
                                                'mean_dist_c_map_mean_f'],
                                    var_name='cat', value_name='value')
                category_mapping = {
                    'mean_dist_c_map_all': 'All points',
                    'mean_dist_c_map_mean_t': 'Observed points',
                    'mean_dist_c_map_mean_f': 'Reconstructed points'
                }
                df_melted['cat'] = df_melted['cat'].map(category_mapping)
                df_melted = df_melted.dropna(subset=['value'])

                plt.figure(figsize=(20, 10))

                sns.boxplot(x='additional_param', y='value', hue='cat', data=df_melted)

                plt.xlabel(r'Variance $\sigma^2$ used in the calculation of the likelihood term of the samples '
                           r'$[\mathrm{mm}^{2}]$')
                plt.ylabel(
                    r'$Average\ distance\ to\ the\ corresponding\ point\ on\ the\ reconstruction\ surface\ (utilising\ '
                    r'given\ fixed_correspondences)\ [\mathrm{mm}]$')
                plt.ylim(0, 40)
                plt.tight_layout()
                filename = f'map_dist_corr_{group}_split.png'
                png_plot_file = self.plot_dir / filename
                plt.savefig(png_plot_file)
                plt.close()

                # Average distance to the closest point across the MAP estimates of all MCMC chains
                plt.figure(figsize=(20, 10))
                sns.boxplot(x='additional_param', y='mean_dist_n_map_all', data=df_group, color='skyblue')
                plt.title(f'Femur reconstruction (LOOCV with N=10) using parallel MCMC sampling (50 chains, MAP sample '
                          f'selected) with an observed portion of {group} per cent')
                plt.xlabel(r'Variance $\sigma^2$ used in the calculation of the likelihood term of the samples '
                           r'$[\mathrm{mm}^{2}]$')
                plt.ylabel(
                    r'$Average\ distance\ to\ the\ closest\ point\ on\ the\ reconstructed\ surface\ [\mathrm{mm}]$')
                plt.ylim(0, 25)
                plt.tight_layout()
                filename = f'map_dist_clp_{group}.png'
                png_plot_file = self.plot_dir / filename
                plt.savefig(png_plot_file)
                plt.close()

                # Squared distances to corresponding points / variances across the MAP estimates of all MCMC chains combined
                fig, ax1 = plt.subplots(figsize=(20, 10))
                mean_of_means_dist = df_group.groupby('additional_param')['mean_dist_c_map_all'].mean().reset_index()
                mean_dist_t = df_group.groupby('additional_param')['mean_dist_c_map_mean_t'].mean().reset_index()
                mean_dist_f = df_group.groupby('additional_param')['mean_dist_c_map_mean_f'].mean().reset_index()
                ax1.scatter(mean_of_means_dist['additional_param'], mean_of_means_dist['mean_dist_c_map_all'],
                            color='red',
                            label='Distances (all)', s=100)
                ax1.scatter(mean_dist_t['additional_param'], mean_dist_t['mean_dist_c_map_mean_t'], color='darkred',
                            label='Distances (observed)', s=100)
                ax1.scatter(mean_dist_f['additional_param'], mean_dist_f['mean_dist_c_map_mean_f'], color='lightcoral',
                            label='Distances (not observed)', s=100)
                plt.plot(mean_of_means_dist['additional_param'], mean_of_means_dist['mean_dist_c_map_all'], color='red')
                plt.plot(mean_of_means_dist['additional_param'], mean_dist_t['mean_dist_c_map_mean_t'], color='darkred')
                plt.plot(mean_of_means_dist['additional_param'], mean_dist_f['mean_dist_c_map_mean_f'],
                         color='lightcoral')
                ax1.set_xlabel(r'Variance $\sigma^2$ used in the calculation of the likelihood term of the samples '
                               r'$[\mathrm{mm}^{2}]$')
                ax1.set_ylabel(
                    r'$Average\ squared\ distance\ to\ the\ corresponding\ point\ on\ the\ reconstruction\ surface\ (utilising\ '
                    r'given\ fixed_correspondences)\ [\mathrm{mm}^{2}]$', color='red')
                ax1.set_ylim(0, 25)
                ax1.legend(loc='upper left')

                ax2 = ax1.twinx()
                mean_of_means_var = df_group.groupby('additional_param')['avg_var_all'].mean().reset_index()
                ax2.scatter(mean_of_means_var['additional_param'], mean_of_means_var['avg_var_all'], color='blue',
                            label='Variances (right)', s=100)
                plt.plot(mean_of_means_var['additional_param'], mean_of_means_var['avg_var_all'], color='blue')
                ax2.set_ylabel(r'$Average\ variance\ of\ the\ points\ [\mathrm{mm}^{2}]$', color='blue')
                ax2.set_ylim(0, 80)
                plt.title(f'Femur reconstruction (LOOCV with N=10) using parallel MCMC sampling (50 chains, MAP sample '
                          f'selected) with an observed portion of {group} per cent')
                fig.tight_layout()
                filename = f'combined_map_dist_corr_var_{group}.png'
                png_plot_file = self.plot_dir / filename
                plt.savefig(png_plot_file)
                plt.close()
