import numpy as np
import torch

from enum import Enum


class ParameterProposalType(Enum):
    MODEL = 0
    TRANSLATION = 1
    ROTATION = 2


class GaussianRandomWalkProposal:

    def __init__(self, batch_size, starting_parameters, sigma_mod=0.1, sigma_trans=1.0, sigma_rot=0.05, chain_length_step=1000):
        """
        The class is used to draw new values for the theta parameters.
        A whole batch of new suggestions is always generated.
        All parameters are drawn independently of a Gaussian distribution with mean value at the previous parameter
        value and standardised variance sigma_mod.

        :param batch_size: Number of proposals to be generated simultaneously.
        :type batch_size: int
        :param starting_parameters: Start values of the parameters. When new parameters are drawn for the first time,
        these parameters are used for each proposal.
        :type starting_parameters: np.ndarray
        :param sigma_mod: Variance of the Gaussian distribution used.
        :type sigma_mod: float
        """
        self.batch_size = batch_size
        self.parameters = torch.tensor(np.tile(starting_parameters[:, np.newaxis], (1, self.batch_size)))
        self.num_parameters = self.parameters.shape[0]
        self.translation = torch.zeros((3, self.batch_size))
        self.rotation = torch.zeros((3, self.batch_size))
        self.sigma_mod = sigma_mod
        self.sigma_trans = sigma_trans
        self.sigma_rot = sigma_rot

        self.old_parameters = None
        self.old_translation = None
        self.old_rotation = None

        self.chain_length = 0
        self.chain_length_step = chain_length_step
        self.chain = torch.zeros((self.num_parameters + 6, self.batch_size, 0))

    def propose(self, parameter_proposal_type: ParameterProposalType):
        """
        Updates the parameter values using a Gaussian random walk (see class description).
        :return: None
        :rtype: None
        """
        if parameter_proposal_type == ParameterProposalType.MODEL:
            perturbations = torch.randn((self.num_parameters, self.batch_size))
        else:
            perturbations = torch.randn((3, self.batch_size))

        self.old_parameters = self.parameters
        self.old_translation = self.translation
        self.old_rotation = self.rotation

        if parameter_proposal_type == ParameterProposalType.MODEL:
            self.parameters = self.parameters + perturbations * self.sigma_mod
        elif parameter_proposal_type == ParameterProposalType.TRANSLATION:
            self.translation = self.translation + perturbations * self.sigma_trans
        else:
            self.rotation = self.rotation + perturbations * self.sigma_rot
        # self.parameters = torch.zeros((self.num_parameters, self.batch_size))
        # self.parameters = perturbations * self.sigma_mod

    def get_parameters(self):
        """
        Returns the current batch of parameter values.

        :return: A 3D tensor with shape (num_parameters, batch_size)
        :rtype: torch.Tensor
        """
        return self.parameters

    def get_translation_parameters(self):
        return self.translation

    def get_rotation_parameters(self):
        return self.rotation

    def update(self, decider):
        self.update_parameters(decider)
        self.update_chain()

    def update_parameters(self, decider):
        self.parameters = torch.where(decider.unsqueeze(0), self.parameters,
                                      self.old_parameters)
        self.translation = torch.where(decider.unsqueeze(0), self.translation, self.old_translation)
        self.rotation = torch.where(decider.unsqueeze(0), self.rotation, self.old_rotation)
        self.old_parameters = None
        self.old_translation = None
        self.old_rotation = None

    def update_chain(self):
        if self.chain_length % self.chain_length_step == 0:
            self.extend_chain()

        self.chain[:, :, self.chain_length] = torch.cat((self.parameters, self.translation, self.rotation), dim=0)
        self.chain_length += 1

    def extend_chain(self):
        updated_chain = torch.zeros((self.num_parameters + 6, self.batch_size, self.chain_length + self.chain_length_step))
        updated_chain[:, :, :self.chain_length] = self.chain
        self.chain = updated_chain
