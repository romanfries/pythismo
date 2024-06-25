import os
from custom_io.MeshIO import MeshReader



class Main:
    def __init__(self):
        print("Hello, world! This is the main class.")

    def run(self):
        file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.path.join('datasets', 'femur-data', 'project-data', 'meshes'), '0.stl')

        mesh_io = MeshReader(file_path)
        mesh = mesh_io.read_mesh()

        print('Successful')


if __name__ == "__main__":
    main = Main()
    main.run()
