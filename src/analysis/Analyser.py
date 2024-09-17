import torch


def autocorrelation(window, lag=1):
    window_size = window.shape[3]
    mean = window.mean(dim=3, keepdim=True)
    y1, y2 = window[:, :, :, :window_size - lag] - mean, window[:, :, :, lag:] - mean
    autocov = (y1 * y2).mean(dim=3)
    return autocov / window.var(dim=3)


class ChainAnalyser:
    def __init__(self, sampler, proposal, model, observed, mesh_chain=None):
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
            mesh_chain = self.model.get_points_from_parameters(self.param_chain.view(self.num_parameters, -1)[:-6])
            # TODO: Apply the correct rotations and translations to the mesh chain. Currently, the correct mesh chain
            #  must be given as input.
            translations = self.param_chain.view(self.num_parameters, -1)[-6:-3]
            rotations = self.param_chain.view(self.num_parameters, -1)[-3:]
            self.num_points = mesh_chain.shape[0]
            self.mesh_chain = mesh_chain.view(self.num_points, 3, self.batch_size, self.chain_length)

    def detect_burn_in(self, window_size=1000, step_size=10, lag=100, threshold=0.0005):
        # TODO: Revise. Is this method suitable? What are good hyperparameters?
        windows = self.param_chain.unfold(dimension=2, size=window_size, step=step_size)
        num_windows = windows.shape[2]
        window_starts = torch.arange(0, self.chain_length - window_size + 1, step_size, device=self.param_chain.device)
        autocorrelations = autocorrelation(windows, lag)
        mask = (torch.abs(autocorrelations[:, :, 1:] - autocorrelations[:, :, :-1])) < threshold
        indices = torch.arange(num_windows - 1, device=self.param_chain.device).unsqueeze(0).unsqueeze(0).expand(
            self.num_parameters, self.batch_size, -1)
        burn_in_window_end = \
        torch.where(mask, indices, torch.full_like(indices, fill_value=num_windows - 1)).min(dim=2)[0]
        burn_in_end = torch.full((self.num_parameters, self.batch_size), self.chain_length, device=self.param_chain.device)
        valid = burn_in_window_end < num_windows
        burn_in_end[valid] = window_starts[burn_in_window_end[valid]]
        return burn_in_end

    def mean_dist_to_target(self, burn_in=2000):
        residuals_c_view = self.residuals_c[:, :, burn_in:].reshape(self.num_points, -1)
        residuals_n_view = self.residuals_n[:, :, burn_in:].reshape(self.num_points, -1)
        return residuals_c_view.mean(dim=1), residuals_n_view.mean(dim=1)

    def avg_variance_per_point(self, burn_in=2000):
        mesh_chain_view = self.mesh_chain[:, :, :, burn_in:].reshape(self.num_points, 3, -1)
        return mesh_chain_view.var(dim=2).mean(dim=1)

    def posterior_analytics(self, burn_in=2000):
        means, mins, maxs = self.posterior[:, burn_in:].mean(dim=1), torch.min(self.posterior[:, burn_in:], dim=1)[0], torch.max(self.posterior[:, burn_in:], dim=1)[0]
        return means, mins, maxs

    def data_to_json(self, loo, obs):
        mean_dist_c, mean_dist_n = self.mean_dist_to_target()
        avg_var = self.avg_variance_per_point()
        means, mins, maxs = self.posterior_analytics()
        acceptance_ratios = self.sampler.acceptance_ratio()
        data = {
            'description': 'MCMC',
            'int_variables': {
                'description': 'TODO',
                'int_var_1': loo,
                'int_var_2': obs
            },
            'float_tensors_200': {
                'description': 'TODO',
                'tensor_1': mean_dist_c.tolist(),
                'tensor_2': mean_dist_n.tolist(),
                'tensor_3': avg_var.tolist()
            },
            'float_tensors_20': {
                'description': 'TODO',
                'tensor_1': means.tolist(),
                'tensor_2': mins.tolist(),
                'tensor_3': maxs.tolist()
            },
            'floats': {
                'description': 'TODO',
                'float_1': acceptance_ratios[0],
                'float_2': acceptance_ratios[1],
                'float_3': acceptance_ratios[2],
                'float_4': acceptance_ratios[3]
            },
            'bool_tensor_200': {
                'description': 'TODO',
                'tensor': self.observed.tolist()
            }
        }
        return data


