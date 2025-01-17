from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    ב: float
        Average validation score over folds
    """
    perm = np.random.permutation(X.shape[0])
    X = X[perm]
    y = y[perm]
    X_folds = np.array_split(X, cv)
    y_folds = np.array_split(y, cv)

    validation_score = []
    train_score = []
    for k in range(cv):
        x_validation, y_validation = X_folds[k], y_folds[k]
        X_train, y_train = np.concatenate(X_folds[:k] + X_folds[k+1:], axis=0),\
                           np.concatenate(y_folds[:k] + y_folds[k+1:], axis=0)
        estimator.fit(X_train, y_train)
        train_score.append(scoring(estimator.predict(X_train), y_train))
        validation_score.append(scoring(estimator.predict(x_validation), y_validation))

    return np.average(train_score), np.average(validation_score)
