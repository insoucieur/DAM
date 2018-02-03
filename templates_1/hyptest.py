#This code is based on equations from Assignment 1 by Aasa Feragen[1],
#http://stackoverflow.com/questions/176918/finding-the-index-of-an-item-given-a-list-containing-it-in-python (post by TerryA)[2]
from __future__ import division
from scipy.stats import t
import numpy as np
import matplotlib.pyplot as plt
import math

data = np.loadtxt('smoking.txt')

# data is the datamatrix of the smoking dataset, e.g. as obtained by data = numpy.loadtxt('smoking.txt')
# should return True if the null hypothesis is rejected and False otherwise, i.e. return p < 0.05
# data is the datamatrix of the smoking dataset, e.g. as obtained by data = numpy.loadtxt('smoking.txt')
# should return True if the null hypothesis is accepted and False otherwise

def hyptest(data):
    ''' Performs a two-sample t-test to investigate whether the alcohol level really is different in good and bad wines.
    Inputs:
    wine_train: datamatrix of the wine training data set
    wine_trainlabels: vector (numpy array) of the corresponding class labels
    Outputs:
    returns True if null hypothesis is accepted and False otherwise '''
    
    np.random.seed(0) #to remove randomisation effects
    
    age = data[:,0]
    FEV1 = data[:,1]
    height = data[:,2]
    gender = data[:,3]
    smoking_status = data[:,4]
    weight = data[:,5]
    nonsmoker_indices = [i for i, n in enumerate(list(smoking_status)) if n == 0]
    smoker_indices = [i for i, n in enumerate(list(smoking_status)) if n == 1]
    nonsmoker = data[nonsmoker_indices,:]
    smoker = data[smoker_indices,:]
    x = nonsmoker[:,1]
    y = smoker[:,1]
    
    #define variables in equations for t-statistic and for v (the degrees of freedom in t-distribution)
    sx_sqr = np.std(x)**2
    sy_sqr = np.std(y)**2
    nx = len(x)
    ny = len(y)
    nx_sqr = nx**2
    ny_sqr = ny**2
    
    if np.std(x) != 0 and np.std(y) !=0: #to remove error: float division by zero when dividing by standard deviations
        #The following lines of code are based on [1]
        t_stats = ( np.mean(x) - np.mean(y) )/( np.sqrt( (sx_sqr/nx)+(sy_sqr/ny) ) )
        v = ( (sx_sqr/nx + sy_sqr/ny)**2 )/( ( sx_sqr**2/(nx_sqr*(nx-1)) ) + ( sy_sqr**2/(ny_sqr*(ny-1)) ) )
        v_down = math.floor(v) #round down v
        
        p = 2*t.cdf(t_stats, v_down)
        print t_stats, v_down, p
        
        alpha = 0.05 #probability of falsely rejecting the null hypothesis
        return p < alpha