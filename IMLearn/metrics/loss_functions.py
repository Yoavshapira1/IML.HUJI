import numpy as np

def fit_dimensions(y_true, y_pred):
    if y_pred.shape != y_true.shape:
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(y_true.shape[0], 1)
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(y_pred.shape[0], 1)

    return y_true, y_pred

def mean_square_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate MSE loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    MSE of given predictions
    """
    y_true, y_pred = fit_dimensions(y_true, y_pred)
    return ((y_true - y_pred)**2).mean()


def misclassification_error(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True) -> float:
    """
    Calculate misclassification loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values
    normalize: bool, default = True
        Normalize by number of samples or not

    Returns
    -------
    Misclassification of given predictions
    """
    y_true, y_pred = fit_dimensions(y_true, y_pred)
    return np.average(y_true != y_pred) if normalize else np.sum(y_true != y_pred)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Accuracy of given predictions
    """
    y_true, y_pred = fit_dimensions(y_true, y_pred)
    return np.average(y_true == y_pred)


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the cross entropy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Cross entropy of given predictions
    """
    raise NotImplementedError()
