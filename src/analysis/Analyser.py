import warnings

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


def ess(chain, autocorrelations, b_int=100, b_max=2000):
    # TODO: Write docstring.
    chain_length = chain.shape[2]
    N = chain_length - torch.arange(0, b_max + 1, b_int, device=chain.device)
    autocorrelation_sums = torch.sum(autocorrelations, dim=2, keepdim=True)
    return (N / (1 + 2 * autocorrelation_sums)).squeeze()


class ChainAnalyser:
    def __init__(self, sampler, proposal, model, observed, mesh_chain=None):
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

    def detect_burn_in(self, ix=10, max_lag=1000, b_int=100, b_max=2000):
        """
        The method calculates an approximation of the ESS (Effective Sample Size) for the chains with different burn-in
        periods and uses this data to automatically detect the burn-in period.
        According to S. Funk et al., a good optimal burn-in length would be when the ESS has hit its maximum for all
        parameters (see https://sbfnk.github.io/mfiidd/index.html).

        :param ix: Specifies how many of the first main components of the model are included in the calculation.
        :type ix: int
        :param max_lag: Specifies after how many elements the infinite sum of the autocorrelation values in the
        approximate calculation of the ESS is truncated.
        :type max_lag: int
        :param b_int: Interval of the tested burn-in period lengths.
        :type b_int: int
        :param b_max: Maximum of the tested burn-in period lengths.
        :type b_max: int
        :return: Determined burn-in period length.
        :rtype: int
        """
        # TODO: Revise. Is this method suitable? What are good hyperparameters? Often does not provide reliable values
        #  at the moment.
        # See https://emcee.readthedocs.io/en/stable/tutorials/autocorr/ for a more in-depth analysis of Monte Carlo
        # errors. Include cross-correlations?
        autocorrelations = acf(self.param_chain[:ix], max_lag, b_int, b_max)
        ess_values = ess(self.param_chain[:ix], autocorrelations, b_int, b_max)
        starts = torch.arange(0, b_max + 1, b_int, device=self.param_chain.device)
        return starts[torch.max(torch.max(ess_values, dim=2)[1])].item()

    def mean_dist_to_target_post(self, burn_in=2000):
        """
        Calculates the average distance of each point of the target to the sampled instances across all chains and
        samples.

        :param burn_in: Determined burn-in period length. This many samples at the beginning of the chains are ignored
        for the calculation.
        :type burn_in: int
        :return: Tuple with two elements:
        - torch.Tensor: Tensor containing the average distances from the target points to the corresponding point on the
        reconstructed mesh instances by utilising the known correspondences with shape (num_points,).
        - torch.Tensor: Tensor containing the average distances to the closest point on the reconstructed mesh
        instances with shape (num_points,).
        :rtype: tuple
        """
        residuals_c_view = self.residuals_c[:, :, burn_in:].reshape(self.num_points, -1)
        residuals_n_view = self.residuals_n[:, :, burn_in:].reshape(self.num_points, -1)
        return residuals_c_view.mean(dim=1), residuals_n_view.mean(dim=1)

    def mean_dist_to_target_map(self, burn_in=2000):
        """
        Calculates the average distance of each point of the target to the maximum a posteriori (MAP) estimate across
        all chains.

        :param burn_in: Determined burn-in period length. This many samples at the beginning of the chains are ignored
        for the calculation.
        :type burn_in: int
        :return: Tuple with two elements:
        - torch.Tensor: Tensor containing the average distances from the target points to the corresponding point on the
        reconstructed mesh instance by utilising the known correspondences with shape (num_points,).
        - torch.Tensor: Tensor containing the average distances to the closest point on the reconstructed mesh
        instance with shape (num_points,).
        :rtype: tuple
        """
        map_indices = torch.max(self.posterior[:, burn_in:], dim=1)[1]
        return self.residuals_c[:, :, burn_in:][:, torch.arange(self.batch_size), map_indices], self.residuals_n[:, :,
                                                                                                burn_in:][:,
                                                                                                torch.arange(
                                                                                                    self.batch_size),
                                                                                                map_indices]

    def avg_variance_per_point_post(self, burn_in=2000):
        """
        Calculates the variance of all the points in the reconstructed mesh instances across all chains and samples.

        :param burn_in: Determined burn-in period length. This many samples at the beginning of the chains are ignored
        for the calculation.
        :type burn_in: int
        :return: Tensor containing the variances mentioned above.
        :rtype: torch.Tensor
        """
        mesh_chain_view = self.mesh_chain[:, :, :, burn_in:].reshape(self.num_points, 3, -1)
        return mesh_chain_view.var(dim=2).mean(dim=1)

    def avg_variance_per_point_map(self, burn_in=2000):
        """
        Calculates the variance of all the points of the maximum a posteriori (MAP) estimates across all chains.

        :param burn_in: Determined burn-in period length. This many samples at the beginning of the chains are ignored
        for the calculation.
        :type burn_in: int
        :return: Tensor containing the variances mentioned above.
        :rtype: torch.Tensor
        """
        map_indices = torch.max(self.posterior[:, burn_in:], dim=1)[1]
        return self.mesh_chain[:, :, :, burn_in:][:, :, torch.arange(self.batch_size), map_indices].var(dim=2).mean(
            dim=1)

    def posterior_analytics(self, burn_in=2000):
        """
        Calculates/determines the mean, maximum, minimum and variances of the unnormalised log density posterior value
        of all chains/batch elements.

        :param burn_in: Determined burn-in period length. This many samples at the beginning of the chains are ignored
        for the calculation.
        :type burn_in: int
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
        means, mins, maxs, vars = self.posterior[:, burn_in:].mean(dim=1), \
        torch.min(self.posterior[:, burn_in:], dim=1)[0], torch.max(self.posterior[:, burn_in:], dim=1)[
            0], self.posterior[:, burn_in:].var(dim=1)
        return means, mins, maxs, vars

    def data_to_json(self, loo, obs):
        # TODO: Not updated and therefore does not contain all metrics calculated in the class.
        """
        Provides all values calculated in this class in .json format.

        :param loo: Indicator of which mesh instance was reconstructed and accordingly omitted from the calculation of
        the corresponding PDM during the run.
        :type loo: int
        :param obs: Observed share of the mesh in per cent.
        :type obs: int
        :return: String containing the data in .json format.
        :rtype: str
        """
        mean_dist_c, mean_dist_n = self.mean_dist_to_target_post()
        avg_var = self.avg_variance_per_point_post()
        means, mins, maxs = self.posterior_analytics()
        acceptance_ratios = self.sampler.acceptance_ratio()
        data = {
            'description': 'MCMC statistics',
            'identifiers': {
                'description': 'TODO',
                'reconstructed shape': loo,
                'percentage observed': obs
            },
            'accuracy': {
                'description': 'TODO',
                'mean distance per point with correspondences': mean_dist_c.tolist(),
                'mean distance per point without correspondences': mean_dist_n.tolist(),
                'variance per point': avg_var.tolist()
            },
            'unnormalised log density posterior value statistics': {
                'description': 'TODO',
                'mean values of the posterior': means.tolist(),
                'min values of the posterior': mins.tolist(),
                'max values of the posterior': maxs.tolist()
            },
            'acceptance ratios': {
                'description': 'TODO',
                'model parameters': acceptance_ratios[0],
                'translation parameters': acceptance_ratios[1],
                'rotation parameters': acceptance_ratios[2],
                'combined acceptance ratio': acceptance_ratios[3]
            },
            'observed points': {
                'description': 'TODO',
                'bool tensor': self.observed.tolist()
            }
        }
        return data
