import numpy as np
import torch


class GaussianRandomWalkProposal:

    def __init__(self, batch_size, starting_parameters, sigma=5.0):
        """
        The class is used to draw new values for the theta parameters.
        A whole batch of new suggestions is always generated.
        All parameters are drawn independently of a Gaussian distribution with mean value at the previous parameter
        value and standardised variance sigma.

        :param batch_size: Number of proposals to be generated simultaneously.
        :type batch_size: int
        :param starting_parameters: Start values of the parameters. When new parameters are drawn for the first time,
        these parameters are used for each proposal.
        :type starting_parameters: np.ndarray
        :param sigma: Variance of the Gaussian distribution used.
        :type sigma: float
        """
        self.batch_size = batch_size
        self.parameters = torch.tensor(np.tile(starting_parameters[:, np.newaxis], (1, self.batch_size)))
        self.num_parameters = self.parameters.shape[0]
        self.sigma = sigma
        self.old_parameters = None

    def propose(self):
        """
        Updates the parameter values using a Gaussian random walk (see class description).
        :return: None
        :rtype: None
        """
        perturbations = torch.randn((self.num_parameters, self.batch_size))
        self.old_parameters = self.parameters
        self.parameters = self.parameters + perturbations * self.sigma

    def get_parameters(self):
        """
        Returns the current batch of parameter values.

        :return: A 3D tensor with shape (num_parameters, batch_size)
        :rtype: torch.Tensor
        """
        return self.parameters

    def update_parameters(self, decider):
        self.parameters = torch.where(decider.unsqueeze(0), self.parameters,
                                      self.old_parameters)
        self.old_parameters = None
