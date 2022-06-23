import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type
from matplotlib import pyplot as plt
from IMLearn import BaseModule
from sklearn.metrics import roc_curve, auc
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from IMLearn.metrics.loss_functions import misclassification_error as MCE
from IMLearn.model_selection.cross_validate import cross_validate as CV

import plotly.graph_objects as go

def split_train_test_numpy(X: np.array, y: np.array, train_proportion: float = .25):
    X_train, y_train, X_test, y_test = split_train_test(pd.DataFrame(X), pd.Series(y), train_proportion)
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values, weights = [], []

    def callback(solver: GradientDescent, w: np.ndarray, val: np.ndarray, grad: np.ndarray,
                 t: int, eta: float, delta: float):
        values.append(val)
        weights.append(w)

    return callback, values, weights


def fixed_rate_optimize(init, etas, plot=True):
    plt.clf()
    vals_1, ws_1, min_loss_1, vals_2, ws_2, min_loss_2 = [], [], [], [], [], []

    for eta in etas:
        norm_1 = L1(init)
        callback_1, values_1, weights_1 = get_gd_state_recorder_callback()
        gd_1 = GradientDescent(learning_rate=FixedLR(eta), callback=callback_1)
        gd_1.fit(norm_1, np.eye(2), np.array([0, 0]))
        if plot:
            plot_descent_path(type(norm_1), np.array(weights_1), title="Module: %s, LR=%.3f" % ("L1", eta)).show()
        vals_1.append(values_1)
        ws_1.append(weights_1)
        min_loss_1.append(np.min(values_1))

        norm_2 = L2(init)
        callback_2, values_2, weights_2 = get_gd_state_recorder_callback()
        gd_2 = GradientDescent(learning_rate=FixedLR(eta), callback=callback_2)
        gd_2.fit(norm_2, np.eye(2), np.array([0, 0]))
        if plot:
            plot_descent_path(type(norm_2), np.array(weights_2), title="Module: %s, LR=%.3f" % ("L2", eta)).show()
        vals_2.append(values_2)
        ws_2.append(weights_2)
        min_loss_2.append(np.min(values_2))

    return min_loss_1, vals_1, ws_1, min_loss_2, vals_2, ws_2


def fixed_rate_plot_convergence(etas, values_1, weights_1, values_2, weights_2, module):
    plt.clf()
    colors = ['r', 'b', 'g', 'y']
    for i, z in enumerate(zip(etas, colors)):
        eta, color = z[0], z[1]
        if module == "L1":
            plt.plot(range(len(values_1[i])), np.linalg.norm(np.array(weights_1[i]), axis=1), c=color,
                     label="$\eta$ = %.3f" % eta)
            if len(values_1[i]) < 1000:
                x_min = np.argmin(np.linalg.norm(np.array(weights_1[i]), axis=1))
                plt.scatter(x_min, 0, c=color)
                plt.text(x_min - .07, - .07, "t = %d" % (int(x_min)), fontsize=9)

        else:
            plt.plot(range(len(values_2[i])), np.linalg.norm(np.array(weights_2[i]), axis=1), c=color,
                     label="$\eta$ = %.3f" % eta)
            if len(values_2[i]) < 1000:
                x_min = np.argmin(np.linalg.norm(np.array(weights_2[i]), axis=1))
                plt.scatter(x_min, 0, c=color)
                plt.text(x_min - .07, - .07, "t = %d" % (int(x_min)), fontsize=9)

    if module == "L1":
        plt.title("L1, w convergence = $\|\|w\|\|^2$")
    else:
        plt.title("L2, w convergence = $\|\|w\|\|^2$")
    plt.xlabel("Iteration")
    plt.ylabel("$\|\|w\|\|^2$")
    plt.legend()
    plt.show()

def fixed_rate_plot_min_loss(etas, loss_1, loss_2):
    plt.clf()

    loss_1 = np.log(loss_1)     # log-scaling to show low results
    loss_1 = (loss_1 - np.min(loss_1)) / (np.max(loss_1) - np.min(loss_1))  # linear stretch
    loss_2 = np.log(loss_2)  # log-scaling to show low results
    loss_2 = (loss_2 - np.min(loss_2)) / (np.max(loss_2) - np.min(loss_2))  # linear stretch

    barWidth = 0.25
    L1_bar = [x for x in range(len(etas))]
    L2_bar = [x + barWidth for x in range(len(etas))]
    bar_1 = plt.bar(L1_bar, loss_1, color='r', width=barWidth,
                    edgecolor='grey', label='L1')
    bar_2 = plt.bar(L2_bar, loss_2, color='b', width=barWidth,
                    edgecolor='grey', label='L2')

    plt.bar_label(bar_1)
    plt.bar_label(bar_2)
    plt.xticks([r + barWidth for r in range(len(etas))], etas)
    plt.title("Minimal loss of L1 & L2 (Fixed rate)")
    plt.xlabel("$\eta$")
    plt.yticks([])
    plt.ylabel("logscale of Loss")
    plt.legend()
    plt.show()

def expo_decay_rate_optimize(init, eta, gammas, plot=True):
    plt.clf()
    vals, ws, min_loss = [], [], []

    for gamma in gammas:
        norm_1 = L1(init)
        callback_1, values_1, weights_1 = get_gd_state_recorder_callback()
        gd = GradientDescent(learning_rate=ExponentialLR(eta, gamma), callback=callback_1)
        gd.fit(norm_1, np.eye(2), np.array([0, 0]))
        if plot:
            plot_descent_path(type(norm_1), np.array(weights_1), title="$\gamma$ = %.3f" % gamma).show()
        vals.append(values_1)
        ws.append(weights_1)
        min_loss.append(np.min(values_1))

    return min_loss, vals, ws

def expo_decay_rate_plot_convergence(gammas, values, weights):
    plt.clf()
    colors = ['r', 'b', 'g', 'y']
    for i, z in enumerate(zip(gammas, colors)):
        eta, color = z[0], z[1]
        plt.plot(range(len(values[i])), np.linalg.norm(np.array(weights[i]), axis=1), c=color,
                 label="$\gamma$ = %.3f" % eta)
        if len(weights[i]) < 1000:
            x_min = np.argmin(np.linalg.norm(np.array(weights[i]), axis=1))
            y_min = np.min(np.linalg.norm(np.array(weights[i]), axis=1))
            plt.scatter(x_min, y_min, c=color)
            plt.text(x_min - .07, y_min - .07, "t = %d" % (int(x_min)), fontsize=9)

    plt.title("L1, expo-decay rate, w convergence = $\|\|w\|\|^2$")
    plt.xlabel("Iteration")
    plt.ylabel("$\|\|w\|\|^2$")
    plt.legend()
    plt.show()

def expo_dacey_rate_plot_min_loss(gammas, loss):
    plt.clf()

    loss = np.log(loss)  # log-scaling to show low results
    loss = (loss - np.min(loss)) / (np.max(loss) - np.min(loss))  # linear stretch

    barWidth = 0.25
    L1_bar = [x for x in range(len(gammas))]
    bar = plt.bar(L1_bar, loss, color='r', width=barWidth,
                    edgecolor='grey')

    plt.bar_label(bar)
    plt.xticks([r for r in range(len(gammas))], gammas)
    plt.xlabel("$\gamma$")
    plt.title("Minimal loss for L1 with exponential decay rate")
    plt.yticks([])
    plt.ylabel("logscale of Loss")
    plt.show()

def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    # Q1
    min_loss_1, vals_1, ws_1, min_loss_2, vals_2, ws_2 = fixed_rate_optimize(init, etas, plot=False)

    # Q2
    fixed_rate_plot_convergence(etas, vals_1, ws_1, vals_2, ws_2, "L1")
    fixed_rate_plot_convergence(etas, vals_1, ws_1, vals_2, ws_2, "L2")

    # Q3
    fixed_rate_plot_min_loss(etas, min_loss_1, min_loss_2)

def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1.)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    min_loss, vals, ws = expo_decay_rate_optimize(init, eta, gammas, plot=False)

    # Plot algorithm's convergence for the different values of gamma
    expo_decay_rate_plot_convergence(gammas, vals, ws)
    expo_dacey_rate_plot_min_loss(gammas, min_loss)

    # Plot descent path for gamma=0.95
    expo_decay_rate_optimize(init, eta, gammas=(.95,), plot=False)


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8):
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test_numpy(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)

def plot_ROC(y, y_prob):
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False, marker_size=5,
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))).show()
    return thresholds[np.argmax(tpr - fpr)]

def Q9(X_train, y_train, X_test, y_test):
    # Plotting convergence rate of logistic regression over SA heart disease data
    logistic_reg = LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000)).fit(X_train, y_train)
    prob = logistic_reg.predict_proba(X_train)
    best_a = plot_ROC(y_train, prob)
    best_reg = LogisticRegression(alpha=best_a).fit(X_train, y_train)
    test_err = best_reg.loss(X_test, y_test)

def Q10_11(X_train, y_train, X_test, y_test, penalty="None"):
    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    val = []
    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    for i in lambdas:
        _, val_score = CV(LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000),
                                             penalty=penalty, alpha=0.5, lam=i), X_train, y_train, MCE)
        val.append(val_score)
    best_l = lambdas[np.argmin(np.array(val))]
    logistic_reg_l1 = LogisticRegression(penalty=penalty, alpha=0.5, lam=best_l).fit(X_train, y_train)

def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Q9
    Q9(X_train, y_train, X_test, y_test)

    # Q10
    Q10_11(X_train, y_train, X_test, y_test, "l1")

    # Q11
    Q10_11(X_train, y_train, X_test, y_test, "l2")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()

