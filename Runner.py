import numpy as np
from matplotlib import pyplot as plt
from IMLearn.learners.regressors import PolynomialFitting as Poly
from IMLearn.metrics import mean_square_error as MSE
import pandas as pd
from exercises import house_price_prediction as hp

if __name__ == "__main__":
    # labels = np.random.randint(0, 15, size=(12,))
    # labels[1], labels[5], labels[10], labels [2], labels[3], labels[6], labels [9] = 0, 0, 0, 0, 0, 0, 0
    # classes, inv = np.unique(labels, return_inverse=True)
    # means = np.random.randint(0,100, size=(len(classes),2))
    # b = means[inv]
    # print("********\n\n\n")
    # print("labels: ", labels)
    # print("means: ", means)
    # print("")
    # print("classes: ", classes)
    # print("inv: ", inv)
    # print(b)
    from matplotlib.lines import Line2D

    # fig, ax = plt.subplots(1, 2)
    # ax[0].scatter(names[:,0], names[:,1], c=0.5, marker='^')
    # plt.show()

    n = 5
    # names = np.random.randint(0,50, size=(5,))
    # data = np.random.randint(0,5, size=(15,))
    # markers = list(Line2D.markers.keys())[2:]
    # colors = np.linspace(0, 1, n)
    # z = zip(markers,colors)
    # dictionary = dict(zip(names, z))

    x = np.arange(6).reshape((3,2))+1
    var = np.array([[[1,0],
                     [0,1]], [[2,0],
                     [0,2]], [[3,0],
                     [0,3]]])
    print(x.shape)
    print(x)
    print(x.shape)
    print(var.shape)
    print('********')
    a = []
    for m, s in zip(x, var):
        a.append(m.T @ s @ m)
    print(a)





