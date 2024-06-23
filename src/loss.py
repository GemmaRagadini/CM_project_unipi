import numpy as np
import sys

def least_squared(y_true, y_pred):
    return np.mean(np.square(np.subtract(y_true,y_pred)))

def least_squared_gradient(y_true, y_pred):  
    return -2*(y_true - y_pred)

def derivative(function):
    if function == least_squared:
        return least_squared_gradient
