import numpy as np
from matplotlib import pyplot as plt
from IMLearn.learners.classifiers import gaussian_naive_bayes as GAU
from IMLearn.learners.regressors import PolynomialFitting as Poly
from IMLearn.metrics import mean_square_error as MSE
import pandas as pd
from exercises import house_price_prediction as hp

if __name__ == "__main__":
    ##### MODEL SELECTION
    # estimator = GAU.GaussianNaiveBayes()
    # cv = 8
    # X = np.random.randint(0,10, size=(100000,3))
    # n, d = X.shape
    # y = np.random.randint(0, 2, size=(n,))
    # print(X,y)
    # score = []
    # X_folds = np.array_split(X, cv)
    # y_folds = np.array_split(y, cv)
    # for k in range(cv):
    #     X_test, y_test = X_folds[k], y_folds[k]
    #     X_train, y_train = np.concatenate(X_folds[:k] + X_folds[k + 1:], axis=0), \
    #                        np.concatenate(y_folds[:k] + y_folds[k + 1:], axis=0)
    #     estimator.fit(X_train, y_train)
    #     score.append(estimator.loss(X_test, y_test))
    # print(score)
    # print(np.average(score))

    X = np.arange(30).reshape(10,3)
    y = np.arange(10)
    print(X)
    print(y)
    perm = np.random.permutation(X.shape[0])
    X = X[perm]
    y = y[perm]
    print(perm)
    print(X)
    print(y)





