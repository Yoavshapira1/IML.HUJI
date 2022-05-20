from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from ...metrics.loss_functions import misclassification_error as MCL
from numpy.linalg import det, inv


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # save the dimensions for more convenience likelihood function
        self.m, self.d = X.shape

        # defining classes and pi
        self.classes_ = np.unique(y)

        # Initialize empty matrices
        self.mu_ = np.empty(shape=(0, self.d))
        self.vars_ = np.empty(shape=(0, self.d))
        self.pi_ = []

        # estimates the mean vectors for every class and the common covariance matrix
        for c in self.classes_:
            X_c = X[y == c]
            mu = np.mean(X_c, axis=0)
            pi = X_c.shape[0] / X.shape[0]
            cov = np.cov((X_c - mu).T)
            self.pi_ = np.r_[self.pi_, pi]
            self.mu_ = np.r_[self.mu_, mu.reshape(1, self.d)]
            if self.d > 1:
                self.vars_ = np.r_[self.vars_, np.diag(cov).reshape(1, self.d)]
            else:       # to allow 1-d data
                self.vars_ = np.r_[self.vars_, cov.reshape(1,1)]

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        likelihood = self.likelihood(X)
        return self.classes_[np.argmax(likelihood, axis=1)]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        likelihood = np.empty(shape=(X.shape[0], 0))
        for i in range(self.classes_.shape[0]):
            prior = np.log(self.pi_[i])
            cov = np.diag(self.vars_[i])

            # dividing vars_[i] by 2 <==> multiply by the inverse matrix of cov
            prob = - 0.5 * np.log(det(cov)) - 0.5 * np.sum(np.power(X - self.mu_[i], 2) / self.vars_[i], 1)
            likelihood = np.c_[likelihood, prior + prob]

        return likelihood

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return MCL(self.predict(X), y)

    def classes_mean_cov(self):
        """Return a list of the covariance matrix of each class"""
        return self.mu_, [np.diag(c) for c in self.vars_]
