import numpy as np


class Regularizer:

    def __init__(self, lamda):
        self.lamda = lamda
    def __call__(self, weights):
        raise NotImplementedError
    def gradient(self, weights):
        raise NotImplementedError


class L1(Regularizer):

    def __call__(self, weights):
        return 0.5 * self.lamda * np.sqrt(np.sum(np.abs(weights)))
    def gradient(self, weights):
        return 0.5 * self.lamda * np.sign(weights)


class L2(Regularizer):

    def __call__(self, weights):
        return 0.5 * self.lamda * np.sqrt(np.sum(np.sum(weights**2)))
    def gradient(self, weights):
        return 0.5 * self.lamda * (weights / (np.sqrt(np.sum(weights**2))))