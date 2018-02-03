import numpy as np
import matplotlib.pyplot as plt
import math

train_data = np.loadtxt('parkinsonsTrain.dt')
test_data = np.loadtxt('parkinsonsTest.dt')
train = train_data[:,0:22]
test = test_data[:,0:22]

# input: 1) train: train data in the form of a N by d numpy array, where N is the number of training data points and d is the number of dimensions
#        2) test: test data in the form of a M by d numpy array, where M is the number of test data points and d is the number of dimensions    
# output: 1) centered and normalized train data as a numpy array
#         2) centered and normalized test data as a numpy array 
def cent_and_norm(train, test):
     #this part is from textbook
    def dot(v, w):
        """v_1 * w_1 + ... + v_n * w_n"""
        return sum(v_i * w_i for v_i, w_i in zip(v, w))
    def sum_of_squares(v):
        """v_1 * v_1 + ... + v_n * v_n"""
        return dot(v, v)
    def de_mean(x):
            """translate x by subtracting its mean (so the result has mean 0)"""
            x_bar = np.mean(x)
            return [x_i - x_bar for x_i in x]
    def variance(x):
        """assumes x has at least two elements"""
        n = len(x)
        deviations = de_mean(x)
        return sum_of_squares(deviations) / (n)
    def std_dv(x):
        return math.sqrt(variance(x))
    #--------------------------------
    
    #find mean of each column and subtract a matrix using these columns with same dimensions as train and data to centre
    def mean_cols(M):
        r, d = M.shape
        mean_cols = []
        for i in range(0,d):
            mean_cols.append(np.mean(M[:,i]))
        return np.array(mean_cols)

    #repeat with variance
    def var_cols(M):
        r, d = M.shape
        var_cols = []
        for i in range(0,d):
            var_cols.append(std_dv(M[:,i]))
        return np.array(var_cols)
    
    def std_matrix(A):
        r, d = A.shape
        mean_train = [mean_cols(train),]*r
        var_train = [var_cols(train),]*r
        return (A - mean_train)/var_train
    h,v = test.shape
    mean_test = [mean_cols(test),]*h
    var_test = [mean_cols(test),]*h
    std_train = std_matrix(train)
    std_test = std_matrix(test)
    
    return std_train, std_test
    pass