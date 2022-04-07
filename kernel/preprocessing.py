"""Preprocessing functions to load the data.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

SHAPE: Tuple[int, int] = (32, 32)
XTR_NAME: str = 'Xtr.csv'
YTR_NAME: str = 'Ytr.csv'
XTE_NAME: str = 'Xte.csv'


def _load_raw_images(path: str) -> np.ndarray:
    """Load the raw images.

    Parameters
    ----------
    path : str
        Full to the csv containing the images.

    Returns
    -------
    np.ndarray
        Images, of shape (num_img, SHAPE[0], SHAPE[1], 3).
    """
    num_pixels = SHAPE[0]*SHAPE[1]
    X = np.array(pd.read_csv(path, header=None,
                 sep=',', usecols=range(3*num_pixels)))
    num_img = X.shape[0]
    assert X.shape[1] == 3*num_pixels
    X_new = np.zeros((num_img, *SHAPE, 3))
    for i in range(num_img):
        X_new[i, :, :, 0] = X[i, :num_pixels].reshape(SHAPE)
        X_new[i, :, :, 1] = X[i, num_pixels:2*num_pixels].reshape(SHAPE)
        X_new[i, :, :, 2] = X[i, 2*num_pixels:3*num_pixels].reshape(SHAPE)
    return X_new


def _load_labels(path: str) -> np.ndarray:
    """Load the labels.

    Parameters
    ----------
    path : str
        Full path to the labels.

    Returns
    -------
    np.ndarray
        Labels, of shape (num_img,).
    """
    return np.array(pd.read_csv(path, sep=',')['Prediction'])


def load_raw_data(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the images and the training labels.

    Parameters
    ----------
    path : str
        Path to the data directory.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Training images, training labels, test images.
    """
    root = Path(path)
    X = _load_raw_images(root.joinpath(XTR_NAME))
    y = _load_labels(root.joinpath(YTR_NAME))
    Xte = _load_raw_images(root.joinpath(XTE_NAME))
    return X, y, Xte
