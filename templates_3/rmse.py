import numpy as np
import matplotlib.pyplot as plt
import math

# input: 1) x: the independent variable, as a N dimensional vector as a numpy array
#        2) y: the dependent variable, as a N dimensional vector as a numpy array
#        3) alpha: the alpha parameter
#        4) beta: the beta parameter
#
# output: 1) the root mean square error (rmse) 

def rmse(x, y, alpha, beta):
    total_se = np.sum((y_i - (alpha + np.dot(beta,x_i)) )** 2 for x_i, y_i in zip(x, y))
    rmse = np.sqrt( total_se / len(x) )
    return rmse
    pass
