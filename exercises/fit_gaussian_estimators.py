from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():

    # Question 1 - Draw samples and print fitted model
    MLE = UnivariateGaussian()
    X = np.random.normal(10, 1, 1000)
    MLE.fit(X)
    print(MLE.mu_, MLE.var_)

    # Question 2 - Empirically showing sample mean is consistent
    ms = np.arange(0, 1000, 10) + 10
    abs_diff = []
    for samples in ms:
        MLE.fit(X[:samples])
        abs_diff.append(np.abs(MLE.mu_ - 10))

    go.Figure([go.Scatter(x=ms, y=abs_diff, mode='markers+lines', name=r'$\widehat\mu$')],
              layout=go.Layout(title=r"$\text{Distance From Real Expectation As Function Of Number Of Samples}$",
                               xaxis_title="$m\\text{ - number of samples}$",
                               yaxis_title="r$\hat\mu$",
                               height=500)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    sort = np.argsort(X)
    pdf = MLE.pdf(X)
    go.Figure([go.Scatter(x=X[sort], y=pdf[sort], mode='markers+lines', name=r'$\widehat\mu$')],
              layout=go.Layout(title=r"$\text{Estimated Probabilty Of Given Values}$",
                               xaxis_title="$m\\text{ - Drawn Values}$",
                               yaxis_title="r$m\\text{ - Probability}$",
                               height=500)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()

