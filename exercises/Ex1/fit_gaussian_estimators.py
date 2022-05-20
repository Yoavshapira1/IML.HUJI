from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
from matplotlib import pyplot as plt
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
                               xaxis_title="$\\text{ - Drawn Values}$",
                               yaxis_title="r$\\text{ - Probability}$",
                               height=500)).show()


def test_multivariate_gaussian():

    # Question 4 - Draw samples and print fitted model
    miu = np.array([0, 0, 4, 0]).T
    cov = np.array([[1,   0.2, 0, 0.5],
                    [0.2, 2,   0,   0],
                    [0,   0,   1,   0],
                    [0.5, 0,   0,   1],])
    MLE = MultivariateGaussian()
    X = np.random.multivariate_normal(miu, cov, 1000)
    MLE.fit(X)
    print(MLE.mu_, "\n", MLE.cov_)

    # Question 5 - Likelihood evaluation
    f = np.linspace(-10 ,10, 200)
    logL = np.fromfunction(lambda i, j: MLE.log_likelihood(np.array([f[i],0,f[j],0]).T, cov, X), (200,200), dtype=int)
    # Heatmap
    plt.imshow(logL, cmap='hot', interpolation='none', extent=[-10 ,10,-10 ,10])
    plt.title(r"Likelihood map of $\mu_1$ and $\mu_3$"+"\n"+r"(where the real $\mu$ is: [0,0,4,0]$)")
    plt.xlabel(r"$\mu_3$")
    plt.ylabel(r"$\mu_1$")
    plt.show()

    # Question 6 - Maximum likelihood
    idx_y, idx_x = np.unravel_index(logL.argmax(), logL.shape)
    print(f[idx_x], f[idx_y])


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()

