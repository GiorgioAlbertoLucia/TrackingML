%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

import sys
sys.path.append('/Users/glucia/Projects/DeepLearning/TrackingML')
from core.geometry import DetectorModule, VoxelGrid

if __name__ == '__main__':
    
    detector_file = '../../data/detectors.csv'
    voxel_grid = VoxelGrid(detector_file, 150, 2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for _, isubmodule in voxel_grid.submodule_dataset.iterrows():
        voxel_grid.add_hit_to_grid(isubmodule['cx'], isubmodule['cy'], isubmodule['cz'])

    plt.savefig('voxel_grid.png')
