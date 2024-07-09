import numpy as np
import torch


class GaussianRandomWalkProposal:
    def __init__(self, batch_size, starting_parameters, sigma=5.0):
        self.batch_size = batch_size
        self.parameters = torch.tensor(np.tile(starting_parameters[:, np.newaxis], (1, self.batch_size)))
        self.num_parameters = self.parameters.shape[0]
        self.sigma = sigma

    def propose(self):
        perturbations = torch.randn((self.num_parameters, self.batch_size))
        self.parameters = self.parameters + perturbations * self.sigma

        return

    def get_parameters(self):
        return self.parameters

    # def update_mesh(self):
    # for index in range(self.batch_size):
