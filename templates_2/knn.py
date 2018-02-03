from __future__ import division
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import math

train = np.loadtxt('parkinsonsTrain.dt')
test = np.loadtxt('parkinsonsTest.dt')
testlabels = test[:,22]
trainlabels = train[:,22]
train_data = train[:,0:22]
test_data = test[:,0:22]

def knn(train, test, trainlabels, k):
    ''' Trains k-NN algorithm using training set and its corresponding labels to predict labels for the test set.
    
    #The following descriptions are from [3]
    Inputs: 
    train: train data (without labels) in the form of a N by d numpy array, where N is the number of training data points and d is the number of dimensions
    test: test data (without labels) in the form of a M by d numpy array, where M is the number of test data points and d is the number of dimensions
    trainlabels: labels for training data in the form of a N by 1 numpy vector, where N is the number of training data points
    k: parameter representing number of nearest neighbours 
    
    Outputs:
    predicted_labels: vector (numpy array) consisting of the predicted classes for the test data'''
    
    np.random.seed(0) #to remove randomisation effects
    
    #The following function is from [2]
    def majority_vote(labels):
        """assumes that labels are ordered from nearest to farthest"""
        vote_counts = Counter(labels)
        winner, winner_count = vote_counts.most_common(1)[0]
        num_winners = len([count
                            for count in vote_counts.values()
                            if count == winner_count])
        if num_winners == 1:
            return winner # unique winner, so return it
        else:
            return majority_vote(labels[:-1]) # try again without the far    
    
    #The following part is based on [1]
    #compute Euclidean distances of test samples from each training samples
    def distance(data1, data2):
        data1num, dim = data1.shape
        data2num, dim = data2.shape
        dist = np.zeros((data1num, data2num))
        for i in range(data1num): #train
            for j in range(data2num): #test
                dist[i,j] = np.linalg.norm(data1[i,:]-data2[j,:])
        return dist
                
    dist = distance(train,test)
	
    #sort distance in order for each test sample (each column in distance matrix)
    sorted_testDistances = np.argsort(dist, axis = 0) #indices of distances sorted in ascending order
    #cut this off at the k-th nearest label to find the corresponding indices in trainlabels
    k_nearest_labels = trainlabels[sorted_testDistances[:k]]
    #do a voting for each test sample to decide on label
    r, c = k_nearest_labels.shape
    predicted_labels_list = []
    for i in range(0,c):
        predicted_labels_list.append(majority_vote(k_nearest_labels[:,i]))
    predicted_labels = np.array(predicted_labels_list)
    dist = dist.T
    return dist, predicted_labels