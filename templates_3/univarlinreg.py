
import numpy as np
import matplotlib.pyplot as plt
import math

# input: 1) x: the independent variable, as a N dimensional vector as a numpy array
#        2) y: the dependent variable, as a N dimensional vector as a numpy array
#
# output: 1) the alpha parameter
#         2) the beta parameter
def univarlinreg(x,y):
    #following part is from textbook:
    #defining functions for calculations
    def dot(v, w):
            """v_1 * w_1 + ... + v_n * w_n"""
            return sum(v_i * w_i for v_i, w_i in zip(v, w))
    def sum_of_squares(v):
        """v_1 * v_1 + ... + v_n * v_n"""
        return dot(v, v)
    def mean(x):
        return sum(x) / len(x)
    def de_mean(x):
        """translate x by subtracting its mean (so the result has mean 0)"""
        x_bar = mean(x)
        return [x_i - x_bar for x_i in x]
    def variance(x):
        """assumes x has at least two elements"""
        n = len(x)
        deviations = de_mean(x)
        return sum_of_squares(deviations) / (n - 1)
    def std_dv(x):
        return math.sqrt(variance(x))
    def covariance(x,y):
        n = len(x)
        return dot(de_mean(x), de_mean(y))/(n-1)
        
    def corr(x,y):
        stdev_x = std_dv(x)
        stdev_y = std_dv(y)
        if stdev_x > 0 and stdev_y > 0:
            return covariance(x,y)/stdev_x / stdev_y
        else:
            return 0
    
    #univariate linear regression part
    beta = corr(x, y) * std_dv(y)/ std_dv(x)
    alpha = mean(y) - beta*mean(x)
    
    return alpha, beta
    pass