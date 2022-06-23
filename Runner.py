from IMLearn.desent_methods.learning_rate import FixedLR, ExponentialLR
import numpy as np

class A:
    def __init__(self, weights: int = 0):
        self.weights_ = weights

    @property
    def weights(self):
        return self.weights_

    @weights.setter
    def weights(self, w : int):
        self.weights_ = w

if __name__ == "__main__":
    # X = np.random.randint(0, 2, size=(15,))
    # y = np.random.randint(0, 2, size=(15,))
    # l = np.empty(shape=(0,2))
    # print(l)
    # pos_X = X[X==1]
    # pos_y = y[X==1]
    # TPR = np.count_nonzero(np.equal(pos_X, pos_y))
    # FPR = pos_X.shape[0] - TPR
    # p = np.array([TPR, FPR]).reshape(2,1)
    # l = np.r_[l, p.T]
    # print(l)
    # l = np.r_[l, p.T]
    # print(l)
    #

    a = np.arange(8).reshape(4,2)
    print(a)
    print(np.linalg.norm(a))
    print(np.linalg.norm(a, axis=0))
    print(np.linalg.norm(a, axis=1))

