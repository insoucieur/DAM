from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math

data = np.loadtxt('Irisdata.txt')
n, _ = data.shape
#k = 5
#seedIndices = (np.random.permutation(n))[:k]

# input:  	1) datamatrix as loaded by numpy.loadtxt('Irisdata.txt').
#			2) vector (numpy array) of indices for k seed observation.
# output: 	1) numpy array of the resulting cluster centers. 
#			2) vector (numpy array) consisting of the assigned cluster for each observation.
# note the indices are an index to an observation in the datamatrix

def kmeans(data, seedIndices): 
    ''' Performs k-means clustering on the dataset given indices that correspond to the initial cluster centres of the data.
    
    #The following descriptions are from [3]
    Inputs: 
    data: datamatrix of the Iris dataset
    seedIndices: vector (numpy array) of indices for k seed observation
    
    Outputs:
    new_centres: numpy array of the resulting cluster centers
    closest_cluster: vector (numpy array) consisting of the indices corresponding to the assigned cluster for each observation'''

    np.random.seed(0) #to remove randomisation effects
    
    k = len(seedIndices)
    #The following function is from [1]
    #compute Euclidean distances from each cluster centre to each data point
    def distance(data1,data2):
        data1_n, dim = data1.shape
        data2_n, dim = data2.shape
        dist = np.zeros((data1_n, data2_n))
        for i in range(data1_n): 
            for j in range(data2_n): 
                dist[i,j] = np.linalg.norm(data1[i,:] - data2[j,:])
        return dist
    
    def assign_cluster(data, centres):
        dist = distance(data,centres) #calculate distances between cluster centres and data points
        closest = np.argmin(dist, axis = 1) #axis = 1 for closest index of each row
        return closest
    
    def cluster_centres(data, closest_cluster, k):
        new_centres = np.zeros(data[:k,].shape)
        for i in range(k):
            clusters = data[closest_cluster==i,:] #assign each data point to its cluster
            new_centres[i] = np.mean(clusters, axis = 0) #axis = 0 to take the mean of the columns of each cluster
        return np.array(new_centres)
    
    #select initial value of cluster centres
    centres = []
    for i in range (0,k):
        centres.append(data[seedIndices[i],:])
    centres = np.array(centres)
    
    #The following part is based on [2]
    #initialise number of iterations and set initial condition
    t = 0
    convergence = 0
    t_max = 100
    
    closest_cluster = assign_cluster(data, centres)
    new_centres = cluster_centres(data, closest_cluster, k)
    
    while convergence == 0:
        t = t + 1
        centres = new_centres
        closest_cluster = assign_cluster(data, centres)
        new_centres = cluster_centres(data, closest_cluster, k)
        
        if (new_centres - centres).any() == 0:
            convergence = 1
        elif t > t_max:
            convergence = 1

    new_centres = np.array(new_centres)
    closest_cluster = np.array(closest_cluster)
    
    return new_centres, closest_cluster
    pass