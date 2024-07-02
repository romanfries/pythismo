import torch


class GaussianRandomWalkProposal:
    def __init__(self, mesh, proposals=10, scale=1.0, sigma=10.0):
        self.mesh = mesh
        self.starting_points = mesh.tensor_points
        self.num_points = mesh.tensor_points.shape[0]
        self.dimension = mesh.tensor_points.shape[1]
        self.proposals = proposals
        self.scale = scale
        self.sigma = sigma
        self.sequence = mesh.tensor_points
        self.applied = False

    def step(self):
        tensor = torch.tensor(())
        perturbations = self.scale * torch.normal(0.0, self.sigma * tensor.new_ones((self.num_points, self.dimension)))
        perturbed_points = self.starting_points + perturbations

        if self.sequence.dim() == 2:
            self.sequence = torch.stack((self.sequence, perturbed_points), dim=2)
        else:
            self.sequence = torch.cat((self.sequence, perturbed_points[:, :, None]), dim=2)

        if not self.applied:
            self.applied = True

        self.proposals -= 1

        return

    def apply(self):
        while self.proposals > 0:
            self.step()

        return

    def get_mesh_k(self, step):
        return self.mesh.new_mesh_from_transformation(self.sequence[:, :, step])
