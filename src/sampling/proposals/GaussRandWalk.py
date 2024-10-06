import warnings
from enum import Enum

import torch


class ParameterProposalType(Enum):
    MODEL_RANDOM = 0
    MODEL_INFORMED = 1
    TRANSLATION = 2
    ROTATION = 3


class GaussianRandomWalkProposal:

    def __init__(self, batch_size, starting_parameters, dev, var_mod, var_trans, var_rot, prob_mod, prob_trans,
                 prob_rot):
        """
        The class is used to draw new values for the parameters. The class supports three types of parameter: Model
        parameters, translation and rotation. It is designed for batches. All parameters are therefore always generated
        for an entire batch of proposals.
        All parameters are drawn independently of a Gaussian distribution with mean value at the previous parameter
        value and a standardised variance. The variance is defined separately for all 3 types of parameters.

        :param batch_size: Number of proposals to be generated simultaneously.
        :type batch_size: int
        :param starting_parameters: Start values of the model parameters. When new parameters are drawn for the first
        time, these initial values are used as the mean value of the Gaussian distribution (for all the elements of the
        batch). The shape of the tensor is assumed to be (num_model_parameters,).
        :type starting_parameters: torch.Tensor
        :param dev: An object representing the device on which the tensor operations are or will be allocated.
        :type dev: torch.device
        :param var_mod: Variance of the model parameters. The variances are given as a one-dimensional tensor of float
        values.
        :type var_mod: torch.Tensor
        :param var_trans: Variance of the translation parameters. The variances are given as a one-dimensional tensor
        of float values.
        :type var_trans: torch.Tensor
        :param var_rot: Variance of the rotation parameters. The variances are given as a one-dimensional tensor of
        float values.
        :type var_rot: torch.Tensor
        :param prob_mod: One-dimensional tensor of floats with the same length as the variances for the model
        parameters. The i-th value for the variance is selected with a probability equal to the i-th entry in this
        tensor.
        :type prob_mod: torch.Tensor
        :param prob_trans: Same principle as for the model parameters.
        :type prob_trans: torch.Tensor
        :param prob_rot: Same principle as for the model parameters.
        :type prob_rot: torch.Tensor
        """
        self.batch_size = batch_size
        self.dev = dev
        self.ratio_trans_prob = torch.ones(self.batch_size, device=self.dev)
        self.parameters = starting_parameters.unsqueeze(1).expand(-1, self.batch_size).to(self.dev)
        self.num_parameters = self.parameters.size()[0]
        self.translation = torch.zeros((3, self.batch_size), device=self.dev)
        self.rotation = torch.zeros((3, self.batch_size), device=self.dev)
        self.var_mod = var_mod
        self.var_trans = var_trans
        self.var_rot = var_rot
        self.prob_mod = prob_mod
        self.prob_trans = prob_trans
        self.prob_rot = prob_rot

        self.old_parameters = None
        self.old_translation = None
        self.old_rotation = None

        self.chain_length = 0
        self.chain = []
        self.posterior = []

        self.sampling_completed = False

    def propose(self, parameter_proposal_type: ParameterProposalType):
        """
        Updates the parameter values using a Gaussian random walk (see class description).

        :param parameter_proposal_type: Specifies which parameters are to be drawn.
        :type parameter_proposal_type: ParameterProposalType
        """
        if self.sampling_completed:
            warnings.warn("Warning: Sampling has already ended for this proposal instance. No new parameter values can"
                          "be proposed.", UserWarning)
            return

        if parameter_proposal_type == ParameterProposalType.MODEL_RANDOM or parameter_proposal_type == \
                ParameterProposalType.MODEL_INFORMED:
            perturbations = torch.randn((self.num_parameters, self.batch_size), device=self.dev)
        else:
            perturbations = torch.randn((3, self.batch_size), device=self.dev)

        self.old_parameters = self.parameters
        self.old_translation = self.translation
        self.old_rotation = self.rotation

        if parameter_proposal_type == ParameterProposalType.MODEL or parameter_proposal_type == \
                ParameterProposalType.MODEL_INFORMED:
            sigma_mod = self.var_mod[torch.multinomial(self.prob_mod, 1).item()].item()
            self.parameters = self.parameters + perturbations * sigma_mod
        elif parameter_proposal_type == ParameterProposalType.TRANSLATION:
            sigma_trans = self.var_trans[torch.multinomial(self.prob_trans, 1).item()].item()
            self.translation = self.translation + perturbations * sigma_trans
        else:
            sigma_rot = self.var_rot[torch.multinomial(self.prob_rot, 1).item()].item()
            self.rotation = self.rotation + perturbations * sigma_rot

    def get_parameters(self):
        """
        Returns the entire batch of current model parameter values.

        :return: A tensor with shape (num_parameters, batch_size)
        :rtype: torch.Tensor
        """
        return self.parameters

    def get_translation_parameters(self):
        """
        Returns the entire batch of current translation parameter values.

        :return: A tensor with shape (3, batch_size)
        :rtype: torch.Tensor
        """
        return self.translation

    def get_rotation_parameters(self):
        """
        Returns the entire batch of current rotation parameter values.

        :return: A tensor with shape (3, batch_size)
        :rtype: torch.Tensor
        """
        return self.rotation

    def update(self, decider, posterior):
        """
        Method to be called when it is clear whether the new parameter values are to be accepted or the old ones are to
        be restored. Regardless of the decision, the current parameter values and log-density values of the posterior
        are saved in the Markov chain.

        :param decider: Boolean tensor of the shape (batch_size,), which indicates for each element of the batch whether
        the new parameter values were accepted (=True) or rejected (=False).
        :type decider: torch.Tensor
        :param posterior: Tensor with shape (batch_size,) containing the new log-density values of the posterior.
        :type posterior: torch.Tensor
        """
        self.parameters = torch.where(decider.unsqueeze(0), self.parameters, self.old_parameters)
        self.translation = torch.where(decider.unsqueeze(0), self.translation, self.old_translation)
        self.rotation = torch.where(decider.unsqueeze(0), self.rotation, self.old_rotation)
        self.old_parameters = None
        self.old_translation = None
        self.old_rotation = None

        self.chain.append(torch.cat((self.parameters, self.translation, self.rotation), dim=0))
        self.posterior.append(posterior)
        self.chain_length += 1

    def change_device(self, dev):
        """
        Change the device on which the tensor operations are or will be allocated. Only execute between completed
        iterations of the sampling process.

        :param dev: The future device on which the mesh data is to be saved.
        :type dev: torch.device
        """
        if self.dev == dev:
            return
        else:
            self.parameters = self.parameters.to(dev)
            self.translation = self.translation.to(dev)
            self.rotation = self.rotation.to(dev)
            if not self.sampling_completed and self.chain_length > 0:
                self.chain = list(torch.stack(self.chain).to(dev).unbind(dim=0))
                self.posterior = list(torch.stack(self.posterior).to(dev).unbind(dim=0))
            elif self.sampling_completed and self.chain_length > 0:
                self.chain = self.chain.to(dev)
                self.posterior = self.posterior.to(dev)

            self.var_mod = self.var_mod.to(dev)
            self.var_trans = self.var_trans.to(dev)
            self.var_rot = self.var_rot.to(dev)

            self.prob_mod = self.prob_mod.to(dev)
            self.prob_trans = self.prob_trans.to(dev)
            self.prob_rot = self.prob_rot.to(dev)

            self.dev = dev

    def close(self):
        """
        Indicates to the instance that no more new parameter values are to be proposed. Converts the chain with the
        parameter values and the log density posterior values from a list to a tensor, which can then be analysed and
        further processed.
        """
        if self.chain_length > 0:
            self.chain = torch.stack(self.chain).permute(1, 2, 0)
            self.posterior = torch.t(torch.stack(self.posterior))
        self.sampling_completed = True

    def get_dict_param_chain_posterior(self):
        """
        Returns the chain with the parameter values and the log density posterior values as a dictionary, which can then
        be saved on disk.

        :return: Above-mentioned dictionary.
        :rtype: dict
        """
        if self.sampling_completed:
            return {'parameters': self.chain, 'posterior': self.posterior}
        else:
            warnings.warn("Warning: The parameter chain can only be saved once sampling has been completed with this "
                          "proposal instance.", UserWarning)
