import os

import matplotlib.pyplot as plt

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    data = pd.read_csv(filename)

    # Remove entries with no price or negative price
    data.dropna(subset=['price'], inplace=True)
    data.drop(data[data['price'] <= 0].index, inplace = True)

    # Handle 'date' column to avoid strings
    # Results in dropping 1 row from the data
    data['date'].replace("", np.nan)
    data.dropna(subset=['date'], inplace=True)
    data['sale_year'] = data['date'].str[0:4].astype('int')
    data['sale_month'] = data['date'].str[5:6].astype('int')
    data['sale_day'] = data['date'].str[7:8].astype('int')

    # Drop unnecessary features
    data.drop(['id', 'lat', 'long', 'date'], inplace=True, axis=1)

    # Add some beneficial features
    data['bathroom_bedroom_ratio'] = data['bathrooms'] / data['bedrooms']
    data['social_status_living'] = data['sqft_living'] / data['sqft_living15']
    data['social_status_lot'] = data['sqft_lot'] / data['sqft_lot15']

    # Handle all "Na" values by changing them to the mean
    # Also, make sure all values are non-negative, otherwise remove the sample
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    for feature in data:
        data[feature].fillna(data[feature].mean(), inplace=True)
        data.drop(data[data[feature] < 0].index, inplace=True)

    # Separate the data
    prices = data['price']
    data.drop(['price'], inplace=True, axis=1)

    # Handle categorical features
    data = pd.get_dummies(data, columns=['zipcode', 'sale_year',
                         'sale_month', 'sale_day'], dummy_na=False)

    return data, prices

def pearson_corr(X, Y):
    cov = np.cov(X,Y)
    sd_x=np.std(X)
    sd_y=np.std(Y)
    corr_cov = cov / (sd_x*sd_y)
    return corr_cov[0,1]

def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    fig = plt.figure()
    y = np.array(y.astype('float'))
    for feature in X:
        feat = np.array(X[feature].astype('float'))
        corr = pearson_corr(feat, y)
        plt.title('Feature: %s\nPearson corr = %.3f' % (feature, corr))
        plt.ylabel('Price, in millions')
        plt.xlabel(feature)
        plt.scatter(feat, y)
        plt.savefig(output_path + r'\%s.jpg' % feature)
        fig.clear()

if __name__ == '__main__':

    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    data_path = r'..\datasets\house_prices.csv'
    data, prices = load_data(data_path)

    # Question 2 - Feature evaluation with respect to response
    cur_dir = os.getcwd()
    mk_dir = cur_dir+r'\feature_eval'
    if not os.path.exists(mk_dir):
        os.makedirs(mk_dir)
    feature_evaluation(data, prices, mk_dir)

    # Question 3 - Split samples into training- and testing sets.
    X_train, y_train, X_test, y_test = split_train_test(data, prices, 0.25)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    LR = LinearRegression()
    frac = np.arange(10, 101)
    MSE = []
    MSE_std = []
    for p in frac:
        p_loss = []
        for i in range(10):
            x_sample = X_train.sample(frac=p/100)
            y_sample = y_train.loc[x_sample.index]
            LR.fit(np.array(x_sample), np.array(y_sample))
            p_loss.append(LR.loss(np.array(X_test), np.array(y_test)))
        MSE.append(np.array([p_loss]).mean())
        MSE_std.append(np.array([p_loss]).std())

    ci = np.array(MSE_std)*2
    MSE = np.array(MSE)

    plt.title('MSE as a function of % of data')
    plt.xlabel('% of the training data')
    plt.ylabel('MSE with confidence interval')
    plt.plot(frac, MSE)
    plt.fill_between(frac, (MSE-ci), (MSE+ci), color='b', alpha=.1)
    plt.show()


