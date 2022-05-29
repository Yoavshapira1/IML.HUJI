from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error as MSE
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso
from matplotlib import pyplot as plt
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def f(x):
    return (x+3)*(x+2)*(x+1)*(x-1)*(x-2)

def split_train_test_numpy(X: np.array, y: np.array, train_proportion: float = .25):
    X_train, y_train, X_test, y_test = split_train_test(pd.DataFrame(X), pd.Series(y), train_proportion)
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    eps = np.random.normal(0, noise, n_samples)
    X = np.linspace(-1.2, 2, n_samples)
    y = f(X) + eps
    X_train, y_train, X_test, y_test = split_train_test_numpy(X, y, 0.66)
    X_train, X_test = X_train.reshape(X_train.shape[0],), X_test.reshape(X_test.shape[0],)
    plt.scatter(X, f(X), c='black', label='Noiseless Model', alpha=0.8)
    plt.scatter(X_train, y_train, c='r', label='Train Set', alpha=0.8)
    plt.scatter(X_test, y_test, c='b', label='Test Set', alpha=0.8)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    plt.clf()
    tr_score = []
    valid_score = []
    for deg in range(11):
        train_score, validation_score = cross_validate(PolynomialFitting(deg), X_train, y_train, MSE)
        tr_score.append(train_score)
        valid_score.append(validation_score)
    min_valid = np.array([0 for x in range(11)], dtype=np.float)
    min_valid[np.argmin(valid_score)] = np.min(valid_score)
    best_fit = int(np.argmin(valid_score))
    barWidth = 0.25
    tr_bar = [x for x in range(11)]
    val_bar = [x + barWidth for x in range(11)]
    plt.bar(tr_bar, tr_score, color='r', width=barWidth,
            edgecolor='grey', label='Training score')
    plt.bar(val_bar, valid_score, color='g', width=barWidth,
            edgecolor='grey', label='Validation score')
    plt.bar(val_bar, min_valid, color='b', width=barWidth,
            edgecolor='grey', label="Minimal validation MSE\nfound in degree %d" % np.argmin(valid_score))
    plt.xlabel('Polynom Degree')
    plt.ylabel('MSE')
    plt.xticks([r + barWidth for r in range(len(tr_bar))], tr_bar)
    plt.legend()
    plt.show()


    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    poly = PolynomialFitting(best_fit)
    poly.fit(X_train, y_train)
    loss = MSE(poly.predict(X_test), y_test)
    print("degree: ", best_fit)
    print("minimal validation error: ", np.min(valid_score))
    print("test error: ", loss)


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    raise NotImplementedError()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    raise NotImplementedError()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree(n_samples=100, noise=5)
    select_polynomial_degree(n_samples=100, noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
