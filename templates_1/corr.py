
import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt('smoking.txt')
type(data)
import math
from scipy.stats import t

# x and y should be vectors of equal length
# should return their correlation as a number
def corr(x,y):
    #following part is from textbook:
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
    
    stdev_x = std_dv(x)
    stdev_y = std_dv(y)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(x,y)/stdev_x / stdev_y
    else:
        return 0
