from pathlib import Path

import h5py
import numpy as np

from src.model.PointDistribution import PointDistributionModel


def load_model_h5(file):
    """
    returns the important information from a h5 model file in the form of a dictionary of np arrays.
    """

    file = Path(file).resolve()
    with h5py.File(file, 'r') as f:
        # pca basis, vectorized format
        basis = f["model"]["pcaBasis"][:]  # 3nxr
        var = f["model"]["pcaVariance"][:]  # r
        mean = f["model"]["mean"][:]  # 3n

        # reference
        points = f["representer"]["points"][:]  # 3xn
        cells = f["representer"]["cells"][:]  # 3xc int

    return {
        "basis": basis,
        "var": var,
        "std": np.sqrt(var),
        "mean": mean,
        "points": points,
        "cells": cells
    }


class ModelReader:
    def __init__(self, relative_path):
        self.relative_path = Path(relative_path)
        self.model_path = Path.cwd().parent / relative_path
        for file in self.model_path.iterdir():
            if file.suffix == '.h5':
                self.model = load_model_h5(file)
        self.read_model = PointDistributionModel(meshes=None, read_in=True, model=self.model)

    def get_model(self):
        return self.read_model



