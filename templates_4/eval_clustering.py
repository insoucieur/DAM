from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math

#data = np.loadtxt('Irisdata.txt')
#r, d = data.shape

def kmeans(data, seedIndices): 
    k = len(seedIndices)
    print k
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
    
    #for k = 1 which returns mean of data
    centre = []
    for i in range (0,1):
        centre.append(data[seedIndices[0],:])
    centre = np.array(centre)
    closest = assign_cluster(data, centre)
    new_centre = cluster_centres(data, closest, 1)
    print new_centre, closest
    
    #select initial value of cluster centres
    centres =[]
    for i in range (0,k):
        centres.append(data[seedIndices[i],:])
    centres = np.array(centres)
    
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
            #np.array(new_centres[closest_cluster])
    return np.array(new_centres), np.array(closest_cluster)

    pass


def objective_function_kmeans(data,centers,clusterLabels):
    ''' Calculates the k-means objective function value for a given k that is deduced from a matrix of the cluster centres.
    
    #The following descriptions are from [2]
    Inputs: 
    data: datamatrix of the Iris dataset
    centers: numpy array of the cluster centers
    clusterLabels: vector (numpy array) consisting of the indices corresponding to the assigned cluster for each observation.
      
    Outputs:
    E: the k-means objective function value as specified in the exam for the given k'''
    
    np.random.seed(0) #to remove randomisation effects    
    
    clusters = centers[clusterLabels] #mu
    k, _ = centers.shape #get k
    #The following line is based on [1]
    E = np.sum( (np.linalg.norm(x_i - mu_j))**2 for x_i, mu_j in zip(data, clusters)) #for each data point
    return E
    pass

def gap_statistics(Erand,E):
    ''' Calculates the gap statistics value for each k=1...N.
    
    #The following descriptions are from [2]
    Inputs: 
    Erand: vector (numpy array) of objective function values for k=1..N for a random dataset
    E: vector (numpy array) of objective function values for k=1..N for a given dataset
    
    Outputs:
    G: vector (numpy array) of the computed gap statistics for each k=1..N'''
    
    np.random.seed(0) #to remove randomisation effects
    
    #The following line is based on [1]
    G = np.log(Erand) - np.log(E)
    return np.array(G)
    pass

def eval_clustering(data,randomData,initialCenters):
    ''' Evaluates the performance of k-means clustering using the gap statistics value for each k=1...N.
    
    #The following descriptions are from [2]
    Inputs: 
    data: Datamatrix
    randomData: Random datamatrix of same size as input data
    initialCenters: numpy array of length N with initial center indices
    
    Outputs:
    G: vector (numpy array) of the computed gap statistics for each k=1..N'''
    
    np.random.seed(0) #to remove randomisation effects
    
    k_total = len(initialCenters)

    E = []
    for i in range(1,k_total+1):
        centers, clusterLabels = kmeans(data, initialCenters[:i])
        E.append(objective_function_kmeans (data, centers, clusterLabels))
    E = np.array(E)
    
    Erand = []
    for i in range(1,k_total+1):
        centers_rand, clusterLabels_rand = kmeans(randomData, initialCenters[:i])
        Erand.append(objective_function_kmeans (randomData, centers_rand, clusterLabels_rand))
    Erand = np.array(Erand)
    
    G = np.array(gap_statistics(Erand,E))
    
    return G
    pass
