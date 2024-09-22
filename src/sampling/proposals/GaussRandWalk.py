import torch

from enum import Enum


class ParameterProposalType(Enum):
    MODEL = 0
    TRANSLATION = 1
    ROTATION = 2


class GaussianRandomWalkProposal:

    def __init__(self, batch_size, starting_parameters, dev, sigma_mod, sigma_trans, sigma_rot, prob_mod, prob_trans,
                 prob_rot, chain_length_step=1000):
        # TODO: Adjust docstring.
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
        :param sigma_mod: Variance of the model parameters.
        :type sigma_mod: float
        :param sigma_trans: Variance of the translation parameters.
        :type sigma_trans: float
        :param sigma_rot: Variance of the rotation parameters.
        :type sigma_rot: float
        :param chain_length_step: Step size with which the Markov chain is extended. If chain_length_step many
        samples were drawn, the chain must be extended again accordingly.
        :type chain_length_step: int
        """
        self.batch_size = batch_size
        self.dev = dev
        self.ratio_trans_prob = torch.ones(self.batch_size, device=self.dev)
        self.parameters = starting_parameters.unsqueeze(1).expand(-1, self.batch_size).to(self.dev)
        self.num_parameters = self.parameters.size()[0]
        self.translation = torch.zeros((3, self.batch_size), device=self.dev)
        self.rotation = torch.zeros((3, self.batch_size), device=self.dev)
        self.sigma_mod = sigma_mod
        self.sigma_trans = sigma_trans
        self.sigma_rot = sigma_rot
        self.prob_mod = prob_mod
        self.prob_trans = prob_trans
        self.prob_rot = prob_rot

        self.old_parameters = None
        self.old_translation = None
        self.old_rotation = None

        self.chain_length = 0
        self.chain_length_step = chain_length_step
        self.chain = torch.zeros((self.num_parameters + 6, self.batch_size, 0), device=self.dev)
        self.posterior = torch.zeros((self.batch_size, 0), device=self.dev)

    def propose(self, parameter_proposal_type: ParameterProposalType):
        """
        Updates the parameter values using a Gaussian random walk (see class description).

        :param parameter_proposal_type: Specifies which parameters are to be drawn.
        :type parameter_proposal_type: ParameterProposalType
        """
        if parameter_proposal_type == ParameterProposalType.MODEL:
            perturbations = torch.randn((self.num_parameters, self.batch_size), device=self.dev)
        else:
            perturbations = torch.randn((3, self.batch_size), device=self.dev)

        self.old_parameters = self.parameters
        self.old_translation = self.translation
        self.old_rotation = self.rotation

        if parameter_proposal_type == ParameterProposalType.MODEL:
            sigma_mod = self.sigma_mod[torch.multinomial(self.prob_mod, 1).item()].item()
            self.parameters = self.parameters + perturbations * sigma_mod
        elif parameter_proposal_type == ParameterProposalType.TRANSLATION:
            sigma_trans = self.sigma_trans[torch.multinomial(self.prob_trans, 1).item()].item()
            self.translation = self.translation + perturbations * sigma_trans
        else:
            sigma_rot = self.sigma_rot[torch.multinomial(self.prob_rot, 1).item()].item()
            self.rotation = self.rotation + perturbations * sigma_rot
        # self.parameters = torch.zeros((self.num_parameters, self.batch_size))
        # self.parameters = perturbations * self.sigma_mod

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
        :param posterior: Tensor with shape (batch_size,) containing the new log-density values of the posterior
        :type posterior: torch.Tensor
        """
        self.update_parameters(decider)
        self.update_chain(posterior)

    def update_parameters(self, decider):
        """
        Internal method that updates the parameters and the log-density values of the posterior according to the
        information from the passed decider.

        :param decider: Boolean tensor of the shape (batch_size,), which indicates for each element of the batch whether
        the new parameter values were accepted (=True) or rejected (=False).
        :type decider: torch.Tensor
        """
        self.parameters = torch.where(decider.unsqueeze(0), self.parameters,
                                      self.old_parameters)
        self.translation = torch.where(decider.unsqueeze(0), self.translation, self.old_translation)
        self.rotation = torch.where(decider.unsqueeze(0), self.rotation, self.old_rotation)
        self.old_parameters = None
        self.old_translation = None
        self.old_rotation = None

    def update_chain(self, posterior):
        """
        Internal method that appends the current parameters and log-density values of the posterior to the Markov chain.
        """
        if self.chain_length % self.chain_length_step == 0:
            self.extend_chain()

        self.chain[:, :, self.chain_length] = torch.cat((self.parameters, self.translation, self.rotation), dim=0)
        self.posterior[:, self.chain_length] = posterior
        self.chain_length += 1

    def extend_chain(self):
        """
        Internal method that is called when the tensors  self.chain/self.posterior are filled.
        The existing Markov chain data is copied to new, larger tensors, which provide space for additional
        self.chain_length_step chain elements.
        """
        updated_chain = torch.zeros((self.num_parameters + 6, self.batch_size, self.chain_length +
                                     self.chain_length_step), device=self.dev)
        updated_chain[:, :, :self.chain_length] = self.chain
        updated_posterior = torch.zeros((self.batch_size, self.chain_length +
                                         self.chain_length_step), device=self.dev)
        updated_posterior[:, :self.chain_length] = self.posterior
        self.chain = updated_chain
        self.posterior = updated_posterior

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
            self.chain = self.chain.to(dev)
            self.posterior = self.posterior.to(dev)

            self.sigma_mod = self.sigma_mod.to(dev)
            self.sigma_trans = self.sigma_trans.to(dev)
            self.sigma_rot = self.sigma_rot.to(dev)

            self.prob_mod = self.prob_mod.to(dev)
            self.prob_trans = self.prob_trans.to(dev)
            self.prob_rot = self.prob_rot.to(dev)

            self.dev = dev
