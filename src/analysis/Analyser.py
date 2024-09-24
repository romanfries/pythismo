import warnings
import arviz as az
import numpy as np
import torch


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
    def __init__(self, sampler, proposal, model, observed, mesh_chain=None, auto_detect_burn_in=False,
                 default_burn_in=2000):
        # TODO: Write docstring.
        self.sampler = sampler
        self.proposal = proposal
        self.param_chain = proposal.chain  # (num_params, batch_size, chain_length)
        self.residuals_c = torch.stack(sampler.residuals_c, dim=-1)  # (num_points, batch_size, chain_length)
        self.residuals_n = torch.stack(sampler.residuals_n, dim=-1)
        self.posterior = proposal.posterior  # (batch_size, chain_length)
        self.model = model
        self.observed = observed
        self.num_parameters = self.param_chain[:-6].shape[0]
        self.batch_size = self.param_chain.shape[1]
        self.chain_length = self.param_chain.shape[2]
        if mesh_chain is not None:
            self.mesh_chain = torch.stack(mesh_chain, dim=-1)  # (num_points, 3, batch_size, chain_length)
            self.num_points = self.mesh_chain.shape[0]
        else:
            warnings.warn("Warning: Functionality to reconstruct the mesh chain not yet implemented. Complete mesh "
                          "chain must be provided as input.", UserWarning)
            mesh_chain = self.model.get_points_from_parameters(self.param_chain.view(self.num_parameters, -1)[:-6])
            # TODO: Apply the correct rotations and translations to the mesh chain. Currently, the correct mesh chain
            #  must be given as input.
            translations = self.param_chain.view(self.num_parameters, -1)[-6:-3]
            rotations = self.param_chain.view(self.num_parameters, -1)[-3:]
            self.num_points = mesh_chain.shape[0]
            self.mesh_chain = mesh_chain.view(self.num_points, 3, self.batch_size, self.chain_length)
        if auto_detect_burn_in:
            self.burn_in = self.detect_burn_in()
        else:
            self.burn_in = default_burn_in

    def detect_burn_in(self, ix=46, b_int=100, b_max=2000):
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

    def ess(self, ix=52):
        """
        The method calculates an approximation of the ESS (Effective Sample Size) for every parameter. The corresponding
        function from the ArviZ library is used to determine the ESS values. As ArviZ does not offer GPU support, the
        calculation may be slow. The current ArviZ implementation is similar to Stan, which uses Geyer’s initial
        monotone sequence criterion (Geyer, 1992; Geyer, 2011).

        :param ix: Specifies how many of the first main components of the model are included in the calculation.
        :type ix: int
        """
        chain_cpu = self.param_chain[:ix, :, self.burn_in:].permute(1, 2, 0).cpu().numpy()
        return torch.tensor(az.ess(az.convert_to_inference_data(chain_cpu)).x.data.astype(np.float32),
                            device=self.param_chain.device)

    def mean_dist_to_target_post(self):
        """
        Calculates the average distance of each point of the target to the sampled instances across all chains and
        samples.

        :return: Tuple with two elements:
        - torch.Tensor: Tensor containing the average distances from the target points to the corresponding point on the
        reconstructed mesh instances by utilising the known correspondences with shape (num_points,).
        - torch.Tensor: Tensor containing the average distances to the closest point on the reconstructed mesh
        instances with shape (num_points,).
        :rtype: tuple
        """
        residuals_c_view = self.residuals_c[:, :, self.burn_in:].reshape(self.num_points, -1)
        residuals_n_view = self.residuals_n[:, :, self.burn_in:].reshape(self.num_points, -1)
        return residuals_c_view.mean(dim=1), residuals_n_view.mean(dim=1)

    def mean_dist_to_target_map(self):
        """
        Calculates the average distance of each point of the target to the maximum a posteriori (MAP) estimate across
        all chains.

        :return: Tuple with two elements:
        - torch.Tensor: Tensor containing the average distances from the target points to the corresponding point on the
        reconstructed mesh instance by utilising the known correspondences with shape (num_points,).
        - torch.Tensor: Tensor containing the average distances to the closest point on the reconstructed mesh
        instance with shape (num_points,).
        :rtype: tuple
        """
        map_indices = torch.max(self.posterior[:, self.burn_in:], dim=1)[1]
        return self.residuals_c[:, :, self.burn_in:][:, torch.arange(self.batch_size), map_indices].mean(dim=1), \
            self.residuals_n[:, :, self.burn_in:][:, torch.arange(self.batch_size), map_indices].mean(dim=1)

    def avg_variance_per_point_post(self):
        """
        Calculates the variance of all the points in the reconstructed mesh instances across all chains and samples.

        :return: Tensor containing the variances mentioned above.
        :rtype: torch.Tensor
        """
        mesh_chain_view = self.mesh_chain[:, :, :, self.burn_in:].reshape(self.num_points, 3, -1)
        return mesh_chain_view.var(dim=2).mean(dim=1)

    def avg_variance_per_point_map(self):
        """
        Calculates the variance of all the points of the maximum a posteriori (MAP) estimates across all chains.

        :return: Tensor containing the variances mentioned above.
        :rtype: torch.Tensor
        """
        map_indices = torch.max(self.posterior[:, self.burn_in:], dim=1)[1]
        return self.mesh_chain[:, :, :, self.burn_in:][:, :, torch.arange(self.batch_size), map_indices].var(
            dim=2).mean(
            dim=1)

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
        means, mins, maxs, vars = self.posterior[:, self.burn_in:].mean(dim=1), \
            torch.min(self.posterior[:, self.burn_in:], dim=1)[0], torch.max(self.posterior[:, self.burn_in:], dim=1)[
            0], self.posterior[:, self.burn_in:].var(dim=1)
        return means, mins, maxs, vars

    def data_to_json(self, loo, obs):
        """
        Provides all values calculated in this class in .json format.

        :param loo: Indicator of which mesh instance was reconstructed and accordingly omitted from the calculation of
        the corresponding PDM during the run.
        :type loo: int
        :param obs: Observed share of the mesh to reconstruct in per cent.
        :type obs: int
        :return: String containing the data in .json format.
        :rtype: str
        """
        ess = self.ess()
        mean_dist_c_post, mean_dist_n_post = self.mean_dist_to_target_post()
        mean_dist_c_map, mean_dist_n_map = self.mean_dist_to_target_map()
        avg_var_post = self.avg_variance_per_point_post()
        avg_var_map = self.avg_variance_per_point_map()
        means, mins, maxs, vars = self.posterior_analytics()
        acc_par, acc_trans, acc_rot, acc_tot = self.sampler.acceptance_ratio()
        data = {
            'description': 'MCMC statistics',
            'identifiers': {
                'reconstructed_shape': loo,
                'percentage_observed': obs
            },
            'effective_sample_sizes': {
                'ess_per_param': ess.tolist()
            },
            'accuracy': {
                'mean_dist_corr_post': mean_dist_c_post.tolist(),
                'mean_dist_clp_post': mean_dist_n_post.tolist(),
                'mean_dist_corr_map': mean_dist_c_map.tolist(),
                'mean_dist_clp_map': mean_dist_n_map.tolist(),
                'var_post': avg_var_post.tolist(),
                'var_map': avg_var_map.tolist()
            },
            'unnormalised_log_density_posterior': {
                'mean': means.tolist(),
                'min': mins.tolist(),
                'max': maxs.tolist(),
                'var': vars.tolist()
            },
            'acceptance': {
                'model': acc_par,
                'translation': acc_trans,
                'rotation': acc_rot,
                'total': acc_tot
            },
            'observed': {
                'boolean': self.observed.tolist()
            }
        }
        return data
