import pandas as pd
from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from IMLearn.metrics.loss_functions import accuracy as Accuracy
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors as clr
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
pio.templates.default = "simple_white"

# Assuming there are no more than 24 different classes
all_markers = list(Line2D.markers.keys())[10:]

def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load('../datasets/%s' % filename)
    return data[:,:2], data[:,2:]


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """

    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X_train, y_train = load_dataset(f)

        losses = []
        def callback(fit: Perceptron, x: np.ndarray, y: int):
            losses.append(fit.loss(X_train, y_train))

        # Fit Perceptron and record loss in each fit iteration
        Perceptron(callback=callback).fit(X_train, y_train)

        # Plot figure
        plt.plot(np.arange(len(losses)), losses)
        plt.xlabel("Iteration")
        plt.ylabel("% of misclassified")
        plt.title("Percentage of misclassifications in %s set" % n)
        plt.show()


def plot_mean_and_cov(ax, estimator):
    # get the estimated mean and cov
    means, cov = estimator.get_mean_and_cov()

    # plot means
    ax.scatter(means[:, 0], means[:, 1], c='black', marker='x')

    # compute the cov ellipse according to the ellipse equation
    # the eigen vectors of the cov specifies the angle of the ellipse
    # the eigen values of the cov specifies the shape (height and weight)
    nstd = 2
    eig_val, eig_vec = np.linalg.eig(cov)
    angle = np.degrees(np.arctan2(*eig_vec[:, 0][::-1]))
    w, h = 2 * nstd * np.sqrt(eig_val)

    for mu in means:
        ell = Ellipse(xy=mu, width=w, height=h, angle=angle, color='black')
        ell.set_facecolor('none')
        ax.add_artist(ell)
        ax.scatter(mu[0], mu[1], c='black', marker='x')

def plot_predictions(ax, classes, X, y_true, y_pred, mark_dict, cmap):
    for px, py, pred, true in zip(X[:, 0], X[:, 1], y_pred, y_true[:, 0]):
        ax.scatter(px, py, color=cmap.to_rgba(pred), marker=mark_dict[true], alpha=0.5)

    # Legend the TRUE classes types - color and shape
    for l in classes:
        ax.scatter([""] * 3, [""] * 3, color=cmap.to_rgba(l), marker=mark_dict[l], label=l)

    # Fixes the axis
    ax.set_xlim(np.min(X[:, 0] - 0.5), np.max(X[:, 0] + 0.5))
    ax.set_ylim(np.min(X[:, 1] - 0.5), np.max(X[:, 1] + 0.5))

    ax.legend(title="Classes")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

def fit_model_and_predict(model, X, y):
    model.fit(X, y)
    y_pred = model.predict(X)
    acc = Accuracy(y_pred=y_pred, y_true=y)

    return y_pred, acc


def get_classes_color_and_markers(y):
    classes = np.unique(y)
    markers = all_markers[:len(classes)]
    markers_dict = dict(zip(classes, markers))
    norm = clr.Normalize(vmin=np.min(classes), vmax=np.max(classes), clip=True)
    cmap = cm.ScalarMappable(norm=norm)

    return classes, markers_dict, cmap


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    LDA_dis = LDA()
    GAU = GaussianNaiveBayes()

    for f in ["gaussian1.npy", "gaussian2.npy"]:
        fig, ax = plt.subplots(1, 2)

        # Load dataset
        X_train, y_train = load_dataset(f)

        # Generate a cmap and markers dictionary for the unique classes types
        classes, mark_dict, cmap = get_classes_color_and_markers(y_train)

        fig.suptitle('Dataset:  %s' % f, fontsize=16)

        i = 0
        for name, estimator in [("LDA estimator", LDA_dis), ("Gaussian Naive Bayes Estimator", GAU)]:

            # Fit models and predict over training set
            y_pred, acc = fit_model_and_predict(estimator, X_train, y_train)

            # subplot title
            ax[i].set_title("%s. Accuracy: %.2f" % (name, acc), fontsize=10)

            # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
            # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
            from IMLearn.metrics import accuracy
            plot_predictions(ax[i], classes, X_train, y_train, y_pred, mark_dict, cmap)

            # Plot the mean and the variance of the Gaussian
            plot_mean_and_cov(ax[i], estimator)

            i += 1
            plt.show()


if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()
