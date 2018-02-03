from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math

data = np.loadtxt('Irisdata.txt')

# input: Datamatrix as loaded by numpy.loadtxt('Irisdata.txt')
# output: Datamatrix of the projected data onto the two first principal components.


# input: Datamatrix as loaded by numpy.loadtxt('Irisdata.txt')
# output: Datamatrix of the projected data onto the two first principal components.
#from sklearn.preprocessing import StandardScaler
def apply_pca(data):
    ''' Performs PCA on the input dataset and projects every data point onto the first 2 principal components.
    
    #The following descriptions are from [2]
    Inputs: 
    data: datamatrix of the Iris dataset
      
    Outputs:
    new_data: datamatrix of the projected data onto the two first principal components'''

    np.random.seed(0) #to remove randomisation effects
    
    #centre the data
    mean_cols = np.mean(data, axis=0)
    var_cols = np.sqrt(np.var(data, axis = 0))
    r, _ = data.shape
    mean_matrix = np.array([mean_cols]*r)
    var_matrix = np.array([var_cols]*r)
    cent_data = (data - mean_matrix)/var_matrix
    
    #The following part is based on [1]
    matrix = np.cov(cent_data.T) #calculate covariance matrix for d dimensions of data
    evals, evecs = np.linalg.eig(matrix) #eigendecomposition of covariance matrix
    evals_r = np.abs(np.real(evals))
    evecs_r = np.real(evecs) #eigenvectors should all have unit length 1
    #sort in descending order
    idx = np.argsort(evals_r)[::-1]
    evals_desc = evals_r[idx] 
    evecs_desc = evecs_r[:,idx] #
 
    #The following part is based on [3]
    #projection matrix to reduce d dimensional space to 2D subspace by choosing the first 2 PCs
    W = np.hstack((evecs_desc[:,0].reshape(4,1), evecs_desc[:,1].reshape(4,1))) #stack arrays as columns
    
    #project data onto subspace spanned by the first 2 PCs
    new_data = np.dot(cent_data, W)
    
    return new_data
    pass
