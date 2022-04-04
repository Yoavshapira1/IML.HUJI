import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from matplotlib import pyplot as plt
from matplotlib import cm
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    data = pd.read_csv(filename, parse_dates=['Date'])
    data.dropna(inplace=True)
    data.drop(data.loc[data['Temp'] < -70.0].index, inplace=True)
    data['dayOfYear'] = data['Date'].dt.dayofyear
    return data

def Q2(data):
    plt.title("AVG temperatures in TLV")
    plt.xlabel("Day of year")
    plt.ylabel("Temperature")
    data_IL = data.loc[data['Country'] == 'Israel']
    for color, grp in data_IL.groupby('Year'):
        plt.scatter('dayOfYear', 'Temp', data=grp, label=color, alpha=0.5)
        plt.legend(title="Year", loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    plt.clf()
    plt.title("AVG and STD of temperature per month")
    plt.xlabel("Month")
    plt.ylabel("AVG and STD")
    month_dst = data_IL.groupby('Month')['Temp'].agg([np.mean, np.std])
    plt.bar(month_dst.index.tolist(), month_dst['mean'], yerr=month_dst['std'], align='center', alpha=0.7, capsize=10)
    plt.show()

def Q3(data):
    plt.clf()
    plt.title("AVG and STD of temperature per month across countries")
    plt.xlabel("Month")
    plt.ylabel("AVG and STD")
    for color, country in data.groupby('Country'):
        tmp = country.groupby('Month')['Temp'].agg([np.mean, np.std])
        plt.errorbar(tmp.index.tolist(), tmp['mean'], tmp['std'], label=color, alpha=0.8)
    plt.legend(title="Year", loc='lower center')
    plt.show()

def Q4(data):
    data_IL = data.loc[data['Country'] == 'Israel']
    y = data_IL['Temp']
    X = data_IL['dayOfYear']
    X_train, y_train, X_test, y_test = split_train_test(X, y)
    loss_per_k = []
    K = 11
    for k in range(1,K):
        Poly = PolynomialFitting(k)
        Poly.fit(np.array(X_train), np.array(y_train))
        loss = round(Poly.loss(np.array(X_test), np.array(y_test)), 2)
        print(loss)
        loss_per_k.append(np.log(loss))

    plt.clf()
    plt.title("Loss as a function of the degree")
    plt.xlabel("Polynomial Estimator degree")
    plt.ylabel("MSE")
    plt.bar([k for k in range(1,K)], loss_per_k)
    plt.show()

def Q5(data):
    # Separate to test and train
    data_train = data.loc[data['Country'] == 'Israel']
    y_train = data_train['Temp']
    X_train = data_train['dayOfYear']
    data_test = data.loc[data['Country'] != 'Israel'].groupby('Country')

    # Fit for k = 3
    Poly = PolynomialFitting(3)
    Poly.fit(np.array(X_train), np.array(y_train))

    res = {}
    for country in data_test:
        y_test = country[1]['Temp']
        X_test = country[1]['dayOfYear']
        res[country[0]] = Poly.loss(X_test, y_test)

    plt.clf()
    plt.title("Loss over each Country")
    plt.xlabel("Country")
    plt.ylabel("MSE")
    plt.bar(res.keys(), [res[v] for v in res.keys()])
    plt.show()

if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data_path = r'..\datasets\City_Temperature.csv'
    data = load_data(data_path)

    # Question 2 - Exploring data for specific country
    Q2(data)

    # Question 3 - Exploring differences between countries
    Q3(data)

    # Question 4 - Fitting model for different values of `k`
    Q4(data)

    # Question 5 - Evaluating fitted model on different countries
    Q5(data)