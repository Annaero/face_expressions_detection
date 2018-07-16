# -*- coding: utf-8 -*-
"""This module provides utils 
that extract features from facial landmark cordinates.
"""

import numpy as np
from scipy.spatial.distance import pdist


def get_mouthes_only(dataset):
    """Converts dataset of landmark coordinates as features
    to dataset of landmarks that belongs to the mouth.
    In 68 landmarks of dlib landmark extraction, landmarks with 
    numbers from 28 to 68.

    Args:
        dataset (np.array): dataset of facial landmarks

    Returns:
        np.array: dataset of mouth landmarks

    """
    indexes = np.arange(48, 68) #maybe magic values should be moved to the config
    return dataset[:, indexes, :]

def flatten(dataset):
    """Converts dataset of landmark coordinates as features
    to dataset with flatten coordinates

    Args:
        dataset (np.array): dataset with shape (m,n,2)

    Returns:
        np.array: dataset with shape (m,n*2)

    """
    x,y,z = dataset.shape
    return dataset.reshape(x, y*z)

def to_dists_dataset(dataset):
    """Converts dataset of landmark coordinates as features
    to dataset of pairvise landmarks distances using euclidean distanse

    Args:
        dataset (np.array): dataset of facial landmarks

    Returns:
        np.array: dataset of landmarks pairvise distances

    """
    return np.array([_coordinates_to_dists(v) for v in dataset])

def _coordinates_to_dists(coordinates, dist='euclidean'):
    return pdist(coordinates, dist)
