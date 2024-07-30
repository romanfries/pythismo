import math

import numpy as np
import torch
from pytorch3d.loss.point_mesh_distance import point_face_distance
from torch.distributions import Uniform

from src.sampling.proposals.GaussRandWalk import ParameterProposalType


def unnormalised_log_posterior(distances, parameters, translation, rotation, sigma_lm, sigma_prior):
    distances_squared = torch.pow(distances, 2)
    log_likelihoods = 0.5 * torch.sum(sigma_lm * (-distances_squared), dim=0)
    log_prior = 0.5 * torch.sum(-torch.pow(parameters / sigma_prior, 2), dim=0)
    uniform_translation_prior = Uniform(-100, 100)
    rotation = (rotation + math.pi) % (2.0 * math.pi) - math.pi
    log_translation = torch.sum(uniform_translation_prior.log_prob(translation), dim=0)
    uniform_rotation_prior = Uniform(-math.pi, math.pi)
    log_rotation = torch.sum(uniform_rotation_prior.log_prob(rotation), dim=0)
    return log_likelihoods + log_prior + log_translation + log_rotation


class PDMMetropolisSampler:
    def __init__(self, pdm, proposal, batch_mesh, target, correspondences=True, sigma_lm=3.0, sigma_prior=1.0):
        self.model = pdm
        self.proposal = proposal
        self.batch_mesh = batch_mesh
        self.points = self.batch_mesh.tensor_points
        self.target = target
        self.target_points = target.tensor_points
        self.correspondences = correspondences
        self.batch_size = self.proposal.batch_size
        self.sigma_lm = sigma_lm
        self.sigma_prior = sigma_prior
        self.old_posterior = None
        self.posterior = None
        self.determine_quality(ParameterProposalType.MODEL)
        self.accepted = 0
        self.rejected = 0

    def propose(self, parameter_proposal_type: ParameterProposalType):
        self.proposal.propose(parameter_proposal_type)

    def update_mesh(self, parameter_proposal_type: ParameterProposalType):
        if parameter_proposal_type == ParameterProposalType.MODEL:
            reconstructed_points = self.model.get_points_from_parameters(self.proposal.get_parameters().numpy())
            if self.batch_size == 1:
                reconstructed_points = reconstructed_points[:, :, np.newaxis]
            self.batch_mesh.set_points(reconstructed_points, save_old=True)
        elif parameter_proposal_type == ParameterProposalType.TRANSLATION:
            self.batch_mesh.apply_translation(self.proposal.get_translation_parameters().numpy(), save_old=True)
        else:
            self.batch_mesh.apply_rotation(self.proposal.get_rotation_parameters().numpy(), save_old=True)
        self.points = self.batch_mesh.tensor_points

    def determine_quality(self, parameter_proposal_type: ParameterProposalType):
        self.update_mesh(parameter_proposal_type)
        self.old_posterior = self.posterior
        if self.correspondences:
            target_points_expanded = self.target_points.unsqueeze(2).expand(-1, -1, self.proposal.batch_size)
            differences = torch.sub(self.points, target_points_expanded)
            # old draft
            # posterior = unnormalised_posterior(differences, self.proposal.parameters, self.sigma_lm, self.sigma_prior)
            distances = torch.linalg.vector_norm(differences, dim=1)
            posterior = unnormalised_log_posterior(distances, self.proposal.parameters, self.proposal.translation,
                                                   self.proposal.rotation, self.sigma_lm, self.sigma_prior)

        else:
            reference_meshes = self.batch_mesh.to_pytorch3d_meshes()
            target_clouds = self.target.to_pytorch3d_pointclouds(self.batch_size)
            # packed representation for faces
            verts_packed = reference_meshes.verts_packed()
            faces_packed = reference_meshes.faces_packed()
            tris = verts_packed[faces_packed]  # (T, 3, 3)
            tris_first_idx = reference_meshes.mesh_to_faces_packed_first_idx()
            # max_tris = reference_meshes.num_faces_per_mesh().max().item()

            points = target_clouds.points_packed()  # (P, 3)
            points_first_idx = target_clouds.cloud_to_packed_first_idx()
            max_points = target_clouds.num_points_per_cloud().max().item()

            # point to face squared distance: shape (P,)
            # requires dtype=torch.float32
            point_to_face_transposed = point_face_distance(points.float(), points_first_idx, tris.float(),
                                                           tris_first_idx, max_points).double().reshape(
                (-1, int(self.model.num_points)))
            point_to_face = torch.transpose(point_to_face_transposed, 0, 1)

            # Target is a point cloud, reference a mesh.
            posterior = unnormalised_log_posterior(torch.sqrt(point_to_face), self.proposal.parameters,
                                                   self.proposal.translation, self.proposal.rotation, self.sigma_lm,
                                                   self.sigma_prior)

        self.posterior = posterior

    def decide(self):
        # log-ratio!
        ratio = torch.exp(self.posterior - self.old_posterior)
        probabilities = torch.min(ratio, torch.ones_like(ratio))
        randoms = torch.rand(self.batch_size)
        decider = torch.gt(probabilities, randoms)
        # decider = torch.ones(self.batch_size).bool()
        self.posterior = torch.where(decider, self.posterior, self.old_posterior)
        self.proposal.update(decider, self.posterior)
        self.batch_mesh.update_points(decider)
        self.points = self.batch_mesh.tensor_points

        self.accepted += decider.sum().item()
        self.rejected += (self.batch_size - decider.sum().item())

    def acceptance_ratio(self):
        return float(self.accepted) / (self.accepted + self.rejected)
