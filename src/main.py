import os, json
from custom_io.MeshIO import MeshReader
from sampling.proposals.GaussRandWalk import GaussianRandomWalkProposal
from visualization.DashViewer import ProposalVisualizer
from registration.Procrustes import ProcrustesAnalyser


class Main:
    def __init__(self):
        pass

    def run(self):
        mesh_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 os.path.join('datasets', 'femur-data', 'project-data', 'meshes'))
        meshes = []
        for file in os.listdir(mesh_path):
            if file.endswith('.stl') or file.endswith('.ply'):
                mesh_file_path = os.path.join(mesh_path, file)
                mesh_io = MeshReader(mesh_file_path)
                mesh = mesh_io.read_mesh()
                meshes.append(mesh)

        landmark_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     os.path.join('datasets', 'femur-data', 'project-data', 'landmarks'))
        landmarks = []
        for file in os.listdir(landmark_path):
            if file.endswith('.json'):
                landmark_file_path = os.path.join(landmark_path, file)
                # Opening JSON file
                f = open(landmark_file_path)
                landmark = json.load(f)
                landmarks.append(landmark)
                f.close()

        landmark_ref_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     os.path.join('datasets', 'femur-data', 'project-data', 'reference-landmarks'))
        for file in os.listdir(landmark_ref_path):
            if file.endswith('.json'):
                landmark_ref_file_path = os.path.join(landmark_ref_path, file)
                # Opening JSON file
                f = open(landmark_ref_file_path)
                landmark_ref = json.load(f)
                f.close()

        landmark_aligner = ProcrustesAnalyser(landmark_ref, landmarks)
        transforms, _ = landmark_aligner.generalised_procrustes_alignment()

        mesh = meshes[0]

        random_walk = GaussianRandomWalkProposal(mesh)
        random_walk.apply()

        visualizer = ProposalVisualizer(random_walk)
        visualizer.run()

        print('Successful')


if __name__ == "__main__":
    main = Main()
    main.run()
