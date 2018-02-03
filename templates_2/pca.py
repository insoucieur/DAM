
import numpy as np
import matplotlib.pyplot as plt
import math
data = np.loadtxt('shapes.txt')
# input: datamatrix as loaded by numpy.loadtxt('shapes.txt')
# output:  1) the eigenvalues in a vector (numpy array) in descending order
#          2) the unit eigenvectors in a matrix (numpy array) with each column being an eigenvector (in the same order as its associated eigenvalue)
#
# note: make sure the order of the eigenvalues is descending, and the eigenvectors have the same order as their associated eigenvalues
def pca(data):
    matrix = np.cov(data)
    evals, evecs = np.linalg.eig(matrix)
    #from http://stackoverflow.com/questions/8092920/sort-eigenvalues-and-associated-eigenvectors-after-using-numpy-linalg-eig-in-pyt
    idx = evals.argsort()[::-1] #sort in descending order
    evals_desc = evals[idx]
    evecs_desc = evecs[:,idx]
    return evals_desc, evecs_desc
    pass