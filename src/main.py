import os
from custom_io.MeshIO import MeshReader
from sampling.proposals.GaussRandWalk import GaussianRandomWalkProposal
from visualization.DashViewer import ProposalVisualizer


class Main:
    def __init__(self):
        pass

    def run(self):
        file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 os.path.join('datasets', 'femur-data', 'project-data', 'meshes'), '0.stl')

        mesh_io = MeshReader(file_path)
        mesh = mesh_io.read_mesh()

        random_walk = GaussianRandomWalkProposal(mesh)
        random_walk.apply()

        visualizer = ProposalVisualizer(random_walk)
        visualizer.run()

        print('Successful')


if __name__ == "__main__":
    main = Main()
    main.run()
