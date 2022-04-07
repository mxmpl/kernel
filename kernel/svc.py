"""Provide kernel SVM for binary classification and
multiclass classification. The optimization is done with cvxopt.
"""
import cvxopt
import numpy as np

from kernel.base import Base
from kernel.kernels import Kernel


class KernelSVC(Base):
    def __init__(self, C: float, kernel: Kernel, epsilon: float = 1e-3) -> None:
        """SVC for binary classification.

        Parameters
        ----------
        C : float
            Regularization parameter.
        kernel : Kernel
            Kernel used by the algorithm.
        epsilon : float, optional
            Tolerance for stopping criterion, by default 1e-3.
        """
        super().__init__()
        self.C = C
        self.kernel = kernel
        self.epsilon = epsilon
        self.alpha = None
        self.support = None
        self.beta = None
        self.b = None

    @property
    def name(self) -> str:
        return 'svc'

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the SVM according to the training data.

        Parameters
        ----------
        X : np.ndarray
            Training vectors, of shape (`num_samples`, `num_features`).
        y : np.ndarray
            Class labels, of shape (`num_samples`,), with values in {-1, 1}.

        Returns
        -------
        self : object
            Fitted classifier.
        """
        cvxopt.solvers.options['show_progress'] = False
        num_samples = len(y)
        K = self.kernel(X, X)
        D = np.diag(y) @ K @ np.diag(y)
        self.log.debug('Kernel computed.')

        # objective function
        D_cvx = cvxopt.matrix(D)
        p_cvx = cvxopt.matrix(-np.ones(num_samples))
        # inequality constraints
        Ginf = - np.identity(num_samples)
        Gsup = np.identity(num_samples)
        G_cvx = cvxopt.matrix(np.vstack((Ginf, Gsup)))
        hinf = np.zeros(num_samples)
        hsup = self.C * np.ones(num_samples)
        h_cvx = cvxopt.matrix(np.hstack((hinf, hsup)))
        # equality constraint
        A_cvx = cvxopt.matrix(y.T, (1, num_samples))
        b_cvx = cvxopt.matrix(0.0)

        # solve QP
        optRes = cvxopt.solvers.qp(D_cvx, p_cvx, G_cvx, h_cvx, A_cvx, b_cvx)
        self.log.debug('Optimization done.')

        self.alpha = np.ravel(optRes['x'])

        # indices of the support vectors
        supportIndices = self.epsilon < self.alpha
        # matrix with each row corresponding to a support vector
        self.support = X[supportIndices]
        # vector containing the weights defining f
        # f(x) = sum_{i \in supportIndices}beta_i K(x_i, x)
        self.beta = (np.diag(y) @ self.alpha)[supportIndices]
        # offset of the classifier
        self.b = (y - K @ np.diag(y) @ self.alpha)[
            (self.epsilon < self.alpha) * (self.alpha < self.C)].mean()
        return self

    def separating_function(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the separating function at a point x.

        Parameters
        ----------
        X : np.ndarray
            (N x d) matrix of d-dimensional samples

        Returns
        -------
        np.ndarray
            Vector of size N
        """
        return self.kernel(X, self.support) @ self.beta + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict y values in {-1, 1}

        Parameters
        ----------
        X : np.ndarray
            Data vectors, of shape (`num_samples`, `num_features`).

        Returns
        -------
        np.ndarray
            Predicted class, of shape (`num_samples`,).
        """
        d = self.separating_function(X)
        return 2 * (d > 0) - 1


class KernelMultiClassSVC(Base):
    def __init__(self, C: float, kernel: Kernel, epsilon: float = 1e-3,
                 num_classes: int = 10) -> None:
        """SVC for multi-class classification using a one-vs-one approach.

        Parameters
        ----------
        C : float
            Regularization parameter.
        kernel : Kernel
            Kernel used by the algorithm.
        epsilon : float, optional
            Tolerance for stopping criterion, by default 1e-3.
        num_classes : int, optional
            Number of classes, by default 10.
        """
        super().__init__()
        self.C = C
        self.kernel = kernel
        self.epsilon = epsilon
        self.num_classes = num_classes
        self.estimators = [KernelSVC(self.C, self.kernel, self.epsilon)
                           for i in range(self.num_classes) for j in
                           range(i + 1, self.num_classes)]
        self.estimators_indices = [[i, j] for i in range(
            self.num_classes) for j in range(i + 1, self.num_classes)]

    @property
    def name(self) -> str:
        return 'multiclass-svc'

    def set_logger(self, *args):
        super().set_logger(*args)
        for svc in self.estimators:
            svc.set_logger(*args)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the multiclass SVM according to the training data.

        Parameters
        ----------
        X : np.ndarray
            Training vectors, of shape (`num_samples`, `num_features`).
        y : np.ndarray
            Class labels, of shape (`num_samples`,), with values in {-1, 1}.

        Returns
        -------
        self : object
            Fitted classifier.
        """
        self.log.debug('Start fit.')
        for index in range(len(self.estimators)):
            i, j = self.estimators_indices[index]
            condition = np.logical_or(y == i, y == j)
            y_label = np.where(y[condition] == i, 1., -1.)
            self.log.debug(f'Fit for labels {i},{j}.')
            self.estimators[index].fit(X[condition, :], y_label)
        self.log.debug('Done')
        return self

    def separating_function(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the separating function.
        Use the same trick as in scikit-learn to add the confidences to
        the votes.
        https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/multiclass.py#L483-L489

        Parameters
        ----------
        X : np.ndarray
            Data vectors, of shape (`num_samples`, `num_features`).

        Returns
        -------
        np.ndarray
            Decision function of shape (`num_samples`, `num_classes`).
        """
        votes = np.zeros((X.shape[0], self.num_classes))
        confidences = np.zeros((X.shape[0], self.num_classes))
        for index in range(len(self.estimators)):
            i, j = self.estimators_indices[index]
            conf = self.estimators[index].separating_function(X)
            confidences[:, i] += conf
            confidences[:, j] -= conf
            votes[:, i] += np.where(conf > 0, 1, 0)
            votes[:, j] += np.where(conf <= 0, 1, 0)
        transformed_confidences = confidences / (3 * (np.abs(confidences) + 1))
        return votes + transformed_confidences

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict y values

        Parameters
        ----------
        X : np.ndarray
            Data vectors, of shape (`num_samples`, `num_features`).

        Returns
        -------
        np.ndarray
            Predicted class, of shape (`num_samples`,), with values
            in {0, ..., self.num_classes-1}.
        """
        return np.argmax(self.separating_function(X), axis=1)
