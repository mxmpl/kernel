"""Utility functions.
"""
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import mode
from tqdm import tqdm

from kernel.svc import KernelMultiClassSVC


def preds_to_csv(preds: np.ndarray, out: str) -> None:
    """Save the predictions in a csv file, ready to be submitted.

    Parameters
    ----------
    preds : np.ndarray
        Predictions, expected to be of length `len(Xtest)`.
    out : str
        Name of the csv file.
    """
    pd.DataFrame(data={'Id': range(1, len(preds)+1), 'Prediction': preds}
                 ).to_csv(out, index=False)


def vote(models: List[KernelMultiClassSVC], X: np.ndarray, best: int
         ) -> np.ndarray:
    """Simple voting method to aggregate predictions from
    multiple classifiers. If all classifiers disagree on one sample,
    we keep the deicision of the best classifier on the validation set.

    Parameters
    ----------
    preds : List[np.ndarray]
        List of multiclass SVMs.
    X : np.ndarray
        Test set, of shape (`num_samples`, `num_features`).
    best : int
        Index of predictions in `preds` for which the
        corresponding classifier is the best, between 0 and `num_classifiers`-1.

    Returns
    -------
    np.ndarray
        Final predictions, of shape (N,).
    """
    preds = np.array([m.predict(X) for m in tqdm(models)])
    res, count = mode(preds)
    res = res[0]
    if 1 in count:
        for i in np.where(count == 1)[1]:
            res[i] = preds[best][i]
    return res


def aggregate(models: List[KernelMultiClassSVC], X: np.ndarray
              ) -> np.ndarray:
    """Aggregate votes from multiple multiclass classifiers.
    Use confidence instead of raw predictions.

    Parameters
    ----------
    models : List[KernelMultiClassSVC]
        List of multiclass SVMs.
    X : np.ndarray
        Test set, of shape (`num_samples`, `num_features`).

    Returns
    -------
    np.ndarray
        Predictions, of shape (`num_samples`), with values in
        {0, ..., `num_classes`-1}.
    """
    preds = np.array([m.separating_function(X) for m in tqdm(models)])
    return np.argmax(preds.sum(axis=0), axis=1)
