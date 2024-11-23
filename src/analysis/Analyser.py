import arviz as az
import numpy as np
import plotly.graph_objects as go

import torch

import pytorch3d.loss
from src.mesh import get_transformation_matrix_from_euler_angles, get_transformation_matrix_from_rot_and_trans, BatchTorchMesh
from src.model import get_parameters
from src.registration import ProcrustesAnalyser, PCAMode


def acf(chain, max_lag=100, b_int=100, b_max=2000):
    # TODO: Write docstring.
    chain_length = chain.shape[2]
    mean = torch.stack([chain[:, :, start:].mean(dim=2, keepdim=True) for start in torch.arange(0, b_max + 1, b_int)],
                       dim=-1)
    var = torch.stack([chain[:, :, start:].var(dim=2, keepdim=True) for start in torch.arange(0, b_max + 1, b_int)],
                      dim=-1)
    autocov = torch.stack([torch.stack([((chain[:, :, start:chain_length - lag] - mean[:, :, :, idx]) * (
            chain[:, :, start + lag:] - mean[:, :, :, idx])).mean(dim=2) for lag in torch.arange(1, max_lag + 1)],
                                       dim=-1) for idx, start in enumerate(torch.arange(0, b_max + 1, b_int))], dim=-1)
    return autocov / var


class ChainAnalyser:
    def __init__(self, sampler, proposal, model, target, observed, mesh_chain=None, auto_detect_burn_in=False,
                 default_burn_in=2000):
        # TODO: Write docstring.
        # TODO: Introduce a device variable.
        self.sampler = sampler
        self.proposal = proposal
        self.param_chain = proposal.chain  # (num_params, batch_size, chain_length)
        self.residuals_c = torch.stack(sampler.residuals_c, dim=-1)  # (num_points, batch_size, chain_length)
        self.residuals_n = torch.stack(sampler.residuals_n, dim=-1)
        self.posterior = proposal.posterior # (batch_size, chain_length)
        self.target = target
        self.model = model
        self.observed = observed
        self.num_parameters = self.param_chain[:-6].shape[0]
        self.batch_size = self.param_chain.shape[1]
        self.true_batch_size = self.batch_size
        self.chain_length = self.param_chain.shape[2]
        if mesh_chain is not None:
            self.mesh_chain = torch.stack(mesh_chain, dim=-1)  # (num_points, 3, batch_size, chain_length)
            self.num_points = self.mesh_chain.shape[0]
            self.rotation_centre = None
        else:
            self.rotation_centre = self.sampler.batch_mesh.rotation_centre[:, 0]
            mesh_chain = self.model.get_points_from_parameters(
                self.param_chain.reshape(self.num_parameters + 6, -1)[:-6])
            translations = self.param_chain.reshape(self.num_parameters + 6, -1)[-6:-3]
            rotations = self.param_chain.reshape(self.num_parameters + 6, -1)[-3:]
            rotation_matrices = get_transformation_matrix_from_euler_angles(rotations, batched_input=True)
            mesh_chain = mesh_chain + (-self.rotation_centre.unsqueeze(0).unsqueeze(-1))
            additional_cols = torch.ones((mesh_chain.size()[0], 1, mesh_chain.size()[2]), device=mesh_chain.device)
            extended_points = torch.cat((mesh_chain, additional_cols), dim=1)
            mesh_chain = torch.bmm(rotation_matrices.permute(2, 0, 1), extended_points.permute(2, 1, 0))[:, :3,
                         :].permute(2, 1, 0) + (self.rotation_centre.unsqueeze(0).unsqueeze(-1)) + translations
            self.num_points = mesh_chain.shape[0]
            self.mesh_chain = mesh_chain.view(self.num_points, 3, self.batch_size, self.chain_length)
        if auto_detect_burn_in:
            self.burn_in = self.detect_burn_in()
        else:
            self.burn_in = default_burn_in

    def detect_burn_in(self, ix=45, b_int=100, b_max=5000):
        """
        The method calculates an approximation of the ESS (Effective Sample Size) for the chains with different burn-in
        periods and uses this data to automatically detect the burn-in period. The corresponding function from the ArviZ
        library is used to determine the ESS values. As ArviZ does not offer GPU support, the calculation may be slow.
        The current ArviZ implementation is similar to Stan, which uses Geyer’s initial monotone sequence criterion
        (Geyer, 1992; Geyer, 2011).
        According to S. Funk et al., a good optimal burn-in length would be when the ESS has hit its maximum for all
        parameters (see https://sbfnk.github.io/mfiidd/index.html).

        :param ix: Specifies how many of the first main components of the model are included in the calculation.
        :type ix: int
        :param b_int: Interval of the tested burn-in period lengths.
        :type b_int: int
        :param b_max: Maximum of the tested burn-in period lengths.
        :type b_max: int
        :return: Determined burn-in period length.
        :rtype: int
        """
        # See https://emcee.readthedocs.io/en/stable/tutorials/autocorr/ for a more in-depth analysis of Monte Carlo
        # errors.
        chain_cpu = self.param_chain[:ix, :, :].permute(1, 2, 0).cpu().numpy()
        starts = np.arange(0, b_max + 1, b_int)
        ess_values = np.zeros((ix, starts.shape[0]))
        for idx, burn_in in enumerate(starts):
            ess_values[:, idx] = az.ess(az.convert_to_inference_data(chain_cpu[:, burn_in:, :])).x.data
        return starts[np.max(np.argmax(ess_values, axis=1))]

    def rhat(self, ix=45):
        """
        The method calculates the rank normalized R-hat diagnostic for the first 'ix' parameters. The rank normalized
        R-hat diagnostic tests for lack of convergence by comparing the variance between multiple chains to the variance
        within each chain. If convergence has been achieved, the between-chain and within-chain variances should be
        identical.

        :param ix: Specifies how many of the first main components of the model are included in the calculation.
        :type ix: int
        :return: Tensor of size (ix,) containing the R-hat values for the first 'ix' parameters of the current chains.
        :rtype: torch.Tensor
        """
        chain_cpu = self.param_chain[:ix, :, self.burn_in:].permute(1, 2, 0).cpu().numpy()
        return torch.tensor(az.rhat(az.convert_to_inference_data(chain_cpu)).x.values.astype(np.float32),
                            device=self.param_chain.device)

    def ess(self, ix=45):
        """
        The method calculates an approximation of the ESS (Effective Sample Size) for the first 'ix' parameters. The
        corresponding function from the ArviZ library is used to determine the ESS values. As ArviZ does not offer GPU
        support, the calculation may be slow. The current ArviZ implementation is similar to Stan, which uses Geyer’s
        initial monotone sequence criterion (Geyer, 1992; Geyer, 2011).

        :param ix: Specifies how many of the first main components of the model are included in the calculation.
        :type ix: int
        :return: Tensor of size (ix,) containing the ESS values for the first 'ix' parameters of the current chains.
        :rtype: torch.Tensor
        """
        chain_cpu = self.param_chain[:ix, :, self.burn_in:].permute(1, 2, 0).cpu().numpy()
        return torch.tensor(az.ess(az.convert_to_inference_data(chain_cpu)).x.data.astype(np.float32),
                            device=self.param_chain.device)

    def mean_dist_to_target_post(self):
        """
        Calculates the average distance of each point of the target to the instances sampled from the posterior
        distribution across all samples of the independent parallel chains.

        :return: Tuple with four elements:
        - torch.Tensor: Tensor containing the average distances from the target points to the corresponding point on the
        reconstructed mesh instances by utilising the known correspondences with shape (num_points, batch_size).
        - torch.Tensor: Tensor containing the average distances to the closest point on the reconstructed mesh
        instances with shape (num_points, batch_size).
        - torch.Tensor: Tensor containing the average squared distances from the target points to the corresponding
        point on the reconstructed mesh instances by utilising the known correspondences with shape
        (num_points, batch_size).
        - torch.Tensor: Tensor containing the average squared distances to the closest point on the reconstructed mesh
        instances with shape (num_points, batch_size).
        :rtype: tuple
        """
        residuals_c_squared, residuals_n_squared = torch.pow(self.residuals_c[:, :, self.burn_in:], 2), torch.pow(self.residuals_n[:, :, self.burn_in:], 2)
        return self.residuals_c[:, :, self.burn_in:].mean(dim=2), self.residuals_n[:, :, self.burn_in:].mean(dim=2), residuals_c_squared.mean(dim=2), residuals_n_squared.mean(dim=2)

    def mean_dist_to_target_map(self):
        """
        Returns the average distance of each point of the target to the maximum a posteriori (MAP) estimate across
        all samples and independent parallel chains.

        :return: Tuple with two elements:
        - torch.Tensor: Tensor containing the average distances from the target points to the corresponding point on the
        reconstructed mesh instance by utilising the known correspondences with shape (num_points,).
        - torch.Tensor: Tensor containing the average distances to the closest point on the reconstructed mesh
        instance with shape (num_points,).
        :rtype: tuple
        """
        map_indices = (torch.div(torch.argmax(self.posterior[:, self.burn_in:]), (self.chain_length - self.burn_in), rounding_mode="trunc")).item(), (torch.argmax(self.posterior[:, self.burn_in:]) % (self.chain_length - self.burn_in)).item()
        return self.residuals_c[:, :, self.burn_in:][:, map_indices[0], map_indices[1]], self.residuals_n[:, :, self.burn_in:][:, map_indices[0], map_indices[1]], torch.pow(self.residuals_c[:, :, self.burn_in:][:, map_indices[0], map_indices[1]], 2), torch.pow(self.residuals_n[:, :, self.burn_in:][:, map_indices[0], map_indices[1]], 2)

    def hausdorff_distances(self):
        map_indices = (torch.div(torch.argmax(self.posterior[:, self.burn_in:]), (self.chain_length - self.burn_in),
                                 rounding_mode="trunc")).item(), (
                    torch.argmax(self.posterior[:, self.burn_in:]) % (self.chain_length - self.burn_in)).item()
        mesh_clouds = self.mesh_chain[:, :, :, self.burn_in:].reshape((self.num_points, 3, self.true_batch_size * (self.chain_length - self.burn_in))).permute(2, 0, 1)
        target_clouds = self.target.tensor_points.unsqueeze(0).expand((self.true_batch_size * (self.chain_length - self.burn_in), -1, -1))
        hausdorff_distances = torch.sqrt(pytorch3d.loss.chamfer_distance(mesh_clouds, target_clouds, batch_reduction=None, point_reduction="max", norm=2)[0])
        return hausdorff_distances.reshape(self.true_batch_size, -1).mean(dim=1), hausdorff_distances.reshape(self.true_batch_size, -1)[map_indices].item()

    def avg_variance_per_point(self):
        """
        Calculates the variance of all the points in the reconstructed mesh instances across all samples of the
        independent parallel chains.

        :return: Tensor containing the variances mentioned above with shape (num_points, batch_size).
        :rtype: torch.Tensor
        """
        within = self.mesh_chain[:, :, :, self.burn_in:].var(dim=3).mean(dim=1)
        between = (self.chain_length - self.burn_in) * self.mesh_chain[:, :, :, self.burn_in:].mean(dim=3).var(dim=2).mean(dim=1)
        return within, between

    def posterior_analytics(self):
        """
        Calculates/determines the mean, maximum, minimum and variances of the unnormalised log density posterior value
        of all chains/batch elements.

        :return: Tuple with four elements:
        - torch.Tensor: Tensor containing the mean of the unnormalised log density posterior value of all chains/batch
         elements with shape (batch_size,).
        - torch.Tensor: Tensor containing the minimum value of the unnormalised log density posterior value of all
        chains/batch elements with shape (batch_size,).
        - torch.Tensor: Tensor containing the maximum value of the unnormalised log density posterior value of all
        chains/batch elements with shape (batch_size,).
        - torch.Tensor: Tensor containing the variances of the unnormalised log density posterior value of all
        chains/batch elements with shape (batch_size,).
        :rtype: tuple
        """
        means, mins, maxs, vars_ = self.posterior[:, self.burn_in:].mean(dim=1), \
            torch.min(self.posterior[:, self.burn_in:], dim=1)[0], torch.max(self.posterior[:, self.burn_in:], dim=1)[
            0], self.posterior[:, self.burn_in:].var(dim=1)
        return means, mins, maxs, vars_

    def simple_convergence_check(self, threshold_val=1.1):
        """
        Performs a simple convergence check by removing chains whose average value of the log density posterior deviates
        too much. It does not guarantee that the chains that pass the check have converged. But there is a high
        probability that the removed chains have not converged. I have not found any scientific justification for this,
        but empirically, it fulfils its purpose.

        :param threshold_val: Determines how strictly chains are removed.
        :type threshold_val: float
        :return: Boolean tensor of shape (batch_size,) that indicates which chains did not pass the convergence check.
        :rtype: torch.Tensor
        """
        means, _, maxs, _ = self.posterior_analytics()
        not_converged = (means < threshold_val * torch.max(means))
        self.true_batch_size = torch.sum(~not_converged).item()
        self.mesh_chain = self.mesh_chain[:, :, ~not_converged, :]
        self.param_chain = self.param_chain[:, ~not_converged, :]
        self.posterior = self.posterior[~not_converged, :]
        self.residuals_c = self.residuals_c[:, ~not_converged, :]
        self.residuals_n = self.residuals_n[:, ~not_converged, :]
        return not_converged

    def error_analysis(self):
        batched_target = BatchTorchMesh(self.target, 'target', self.proposal.dev, batch_size=1)
        batched_mean = batched_target.copy()
        batched_mean.set_points(self.model.mean.reshape((self.num_points, 3)).unsqueeze(-1))

        map_indices = (torch.div(torch.argmax(self.posterior[:, self.burn_in:]), (self.chain_length - self.burn_in),
                                 rounding_mode="trunc")).item(), (
                torch.argmax(self.posterior[:, self.burn_in:]) % (self.chain_length - self.burn_in)).item()
        map_points = self.mesh_chain[:, :, map_indices[0], map_indices[1]]
        map_params = self.param_chain[:, map_indices[0], map_indices[1]]

        rotation_centre = self.sampler.batch_mesh.rotation_centre[:, 0]
        batched_map = batched_target.copy()
        batched_map.rotation_centre = rotation_centre.unsqueeze(-1)
        batched_map.set_points(self.model.get_points_from_parameters(map_params[:-6]).unsqueeze(-1))
        batched_map.apply_rotation(map_params[-3:].unsqueeze(-1))
        batched_map.apply_translation(map_params[-6:-3].unsqueeze(-1))

        diff = batched_map.tensor_points.squeeze() - map_points

        proc = ProcrustesAnalyser(batched_target, batched_mean, mode=PCAMode.BATCHED)
        rot, trans = proc.procrustes_alignment()
        batched_target.apply_transformation(get_transformation_matrix_from_rot_and_trans(rot.permute(1, 2, 0), trans.permute(1, 0)))

    def data_to_json(self, ix, loo, obs, additional_param):
        """
        Provides all values calculated in this class in .json format.

        :param ix: Specifies for how many of the (model) parameters the MCMC diagnostics ESS and Rhat are to be
        determined.
        :type ix: int
        :param loo: Indicator of which mesh instance was reconstructed and accordingly omitted from the calculation of
        the corresponding PDM during the run.
        :type loo: int
        :param obs: Observed share of the mesh to reconstruct in per cent.
        :type obs: int
        :param additional_param: Additional parameter to distinguish the generated outputs from each other.
        :type additional_param: int
        :return: String containing the data in .json format.
        :rtype: str
        """
        # self.error_analysis()
        not_converged = self.simple_convergence_check()
        # not_converged = torch.zeros((self.batch_size,), dtype=torch.bool, device=self.param_chain.device)
        means, mins, maxs, vars = self.posterior_analytics()
        rhat = self.rhat(ix=ix)
        ess = self.ess(ix=ix)
        mean_dist_c_post, mean_dist_n_post, squared_c_post, squared_n_post = self.mean_dist_to_target_post()
        mean_dist_c_map, mean_dist_n_map, squared_c_map, squared_n_map = self.mean_dist_to_target_map()
        hausdorff_avg, hausdorff_map = self.hausdorff_distances()
        avg_var_within, avg_var_between = self.avg_variance_per_point()
        acc_par, acc_rnd, acc_trans, acc_rot, acc_tot = self.sampler.acceptance_ratio(~not_converged)
        data = {
            'description': 'MCMC statistics',
            'identifiers': {
                'reconstructed_shape': loo,
                'percentage_observed': obs,
                'additional_param': additional_param,
                'chains_considered': torch.sum(~not_converged).item()
            },
            'effective_sample_sizes': {
                'ess_per_param': ess.tolist(),
                'rhat_per_param': rhat.tolist()
            },
            'accuracy': {
                'mean_dist_corr_post': mean_dist_c_post.tolist(),
                'mean_dist_clp_post': mean_dist_n_post.tolist(),
                'squared_corr_post': squared_c_post.tolist(),
                'squared_clp_post': squared_n_post.tolist(),
                'mean_dist_corr_map': mean_dist_c_map.tolist(),
                'mean_dist_clp_map': mean_dist_n_map.tolist(),
                'squared_corr_map': squared_c_map.tolist(),
                'squared_clp_map': squared_n_map.tolist(),
                'hausdorff_avg': hausdorff_avg.tolist(),
                'hausdorff_map': hausdorff_map,
                'var': avg_var_within.tolist(),
                'var_between': avg_var_between.tolist()
            },
            'unnormalised_log_density_posterior': {
                'mean': means.tolist(),
                'min': mins.tolist(),
                'max': maxs.tolist(),
                'var': vars.tolist()
            },
            'acceptance': {
                'model': acc_par,
                'random_noise': acc_rnd,
                'translation': acc_trans,
                'rotation': acc_rot,
                'total': acc_tot
            },
            'observed': {
                'boolean': self.observed.tolist()
            }
        }

        return data

    def get_traceplots(self, loo, obs, additional_param):
        figures = []
        posterior_cpu = self.posterior.cpu().numpy()
        y_min, y_max = np.min(posterior_cpu[:, self.burn_in:]), np.max(posterior_cpu[:, self.burn_in:])
        for i in range(self.posterior.size(0)):
            x, y = np.arange(self.burn_in, self.chain_length, 1), posterior_cpu[i, self.burn_in:self.chain_length]
            figure = {
                'data': [go.Scattergl(x=x, y=y, mode='lines', marker=dict(size=2))],
                'layout': go.Layout(
                    title=f'Trace Plot for Chain {i}',
                    xaxis={'title': 'Iteration'},
                    yaxis={'title': 'Log Density Values', 'range': [y_min, y_max]},
                    annotations=[
                        dict(
                            xref='paper',
                            yref='paper',
                            x=1,
                            y=1.01,  # Position auf der y-Achse (0.95 ist nahe der oberen Kante)
                            showarrow=False,  # Keine Pfeile, nur Text
                            text=f"Reconstructed Shape: {loo}<br>"
                                 f"Observed Length Proportion: {obs}<br>"
                                 f"Variance for Indep. Point Likelihood Evaluator: {additional_param}",
                            align='left',
                            yanchor='bottom',
                            bordercolor='black',
                            borderwidth=1,
                            borderpad=4,
                            bgcolor='lightgrey',
                            font=dict(size=10)
                        )
                    ]
                )
            }
            figures.append(figure)
        return figures





