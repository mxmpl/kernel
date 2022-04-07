"""Provide multiple kernel implementations.
"""
import abc
from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist

from kernel.base import Base


class Kernel(Base, abc.ABC):
    """Kernel base class."""

    @abc.abstractmethod
    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute the kernel.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)

        Y : ndarray of shape (n_samples_Y, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples_X, n_samples_Y)
            Gram matrix.
        """


class LinearKernel(Kernel):
    """Compute the linear kernel between X and Y."""

    @property
    def name(self) -> str:
        return 'linear-kernel'

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute the linear kernel.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)

        Y : ndarray of shape (n_samples_Y, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples_X, n_samples_Y)
            Gram matrix.
        """
        self.log.debug(f'Compute kernel between X {X.shape} and Y {Y.shape}')
        return X @ Y.T


class PolynomialKernel(Kernel):

    def __init__(self, degree: int, gamma: Optional[float] = None,
                 coef0: float = 0.) -> None:
        """Compute the polynomial kernel between X and Y::
            K(X, Y) = (gamma <X, Y> + coef0)^degree

        Parameters
        ----------
        degree : int
            Degree of the polynomial kernel.
        gamma : Optional[float], optional
            Multiplicative factor, by default None
        coef0 : float, optional
            Additive constant, by default 0.
        """
        super().__init__()
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma

    @property
    def name(self) -> str:
        return 'polynomial-kernel'

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute the polynomial kernel.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)

        Y : ndarray of shape (n_samples_Y, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples_X, n_samples_Y)
            Gram matrix.
        """
        if self.gamma is None:
            self.gamma = 1./(X.shape[1] * np.var(X))
        self.log.debug(f'Compute kernel between X {X.shape} and Y {Y.shape},'
                       f'with gamma={self.gamma:0.3e}')
        K = X @ Y.T
        K *= self.gamma
        K += self.coef0
        K **= self.degree
        return K


class RBFKernel(Base):

    def __init__(self, gamma: Optional[float] = None):
        """Compute the rbf (gaussian) kernel between X and Y:
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Parameters
        ----------
        gamma : float, optional
            Multiplicative factor, by default None.
        """
        super().__init__()
        self.gamma = gamma

    @property
    def name(self):
        return 'rbf-kernel'

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute the RBF kernel.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)

        Y : ndarray of shape (n_samples_Y, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples_X, n_samples_Y)
            Gram matrix.
        """
        if self.gamma is None:
            self.gamma = 1./(X.shape[1] * np.var(X))
        self.log.debug(f'Compute kernel between X {X.shape} and Y {Y.shape},'
                       f'with gamma={self.gamma:0.3e}')
        K = cdist(X, Y, metric="sqeuclidean")
        np.exp(-self.gamma * K, K)  # exponentiate K in-place
        return K
