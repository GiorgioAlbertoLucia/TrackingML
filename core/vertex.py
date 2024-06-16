'''
    Clss for vertexing and solid angle calculations
'''

import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN

class Vertexing:

    def __init__(self, dataset: pd.DataFrame):
        '''
            Initialize the vertexing class
            Expects the dataset to be a pd.Dataframe with columns x, y, z and particle_id
        '''

        self.vertex = self._compute_vertex(dataset)

    def _compute_vertex(self, dataset: pd.DataFrame) -> np.ndarray:
        '''
            Compute the vertex of the event
            Expects the dataset to be a pd.Dataframe with columns x, y, z and particle_id
        '''
        particle_df = dataset.query('particle_id != 0')

        hit_positions = particle_df[['x', 'y', 'z']].values
        clustering = DBSCAN(eps=0.01, min_samples=10).fit(hit_positions)

        labels = clustering.labels_
        unique_labels, counts = np.unique(labels, return_counts=True)
        main_cluster_label = unique_labels[np.argmax(counts)]

        main_cluster_hits = hit_positions[labels == main_cluster_label]
        return np.mean(main_cluster_hits, axis=0)

    def evaluate_solid_angle(self, point1, point2) -> float:
        """
        Compute the solid angle that contains two hits and the vertex.
        
        Given two vectors `a` and `b` that connect two points to a common vertex, the angle 
        between them is given by:
        
        θ = arccos((a · b) / (|a| |b|))
        
        Let's consider the spherical surface with radius equal to the highest magnitude of 
        the two vectors. Let's also consider the cone with its axis along the first vector 
        and its side along the other vector, with length equal to the second vector. The 
        solid angle occupied by the cone in the sphere is given by the ratio between the 
        area of the base of the cone (π (|b| sinθ)^2) and the square of the radius of the 
        sphere (|b|^2).
        
        Ω = π sin^2(θ)
        
        Parameters:
        point1 (tuple): A tuple representing the coordinates (x, y, z) of the first point.
        point2 (tuple): A tuple representing the coordinates (x, y, z) of the second point.
        
        Returns:
        float: The solid angle in steradians.
        """
        point1 = np.array(point1)
        point2 = np.array(point2)
        
        vector_a = point1 - self.vertex
        vector_b = point2 - self.vertex
        
        arg = np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
        if arg > 1.:
            arg = 1.
        elif arg < -1.:
            arg = -1.
        theta = np.arccos(arg)
        solid_angle = np.pi * (np.sin(theta) ** 2)
        
        return solid_angle
        