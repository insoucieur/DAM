from __future__ import division
from sklearn.cross_validation import KFold
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import math

train = np.loadtxt('parkinsonsTrain.dt')
trainlabels = train[:,22]
train = train[:,0:22]
rand_perm = np.random.permutation(len(trainlabels))


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

    return predicted_labels

def cv(train, trainlabels, rand_perm):
    ''' Performs 5 fold cross validation on training set to find a good value an optimal value for k when k = 1, 3, ..., 9.
    
    #The following descriptions are from [3]
    Inputs: 
    train: training data in the form of a N by d numpy array, where N is the number of training data points and d is the number of dimensions
    trainlabels: training labels in the form of a N by 1 numpy vector, where N is the number of training data points
    rand_perm: a random permutation of entries as a numpy array, e.g. np.random.permutation(len(trainlabels))
    
    Outputs:
    k_best: the optimal k'''
    
    #np.random.seed(0) #to remove randomisation effects
    
    #The following part is based on [4]
    kf = KFold(len(trainlabels), n_folds=5)
    X = train
    y = trainlabels
    index_list = list(kf)
    
    accuracy_lists = []
    for i in range (0, 5):
        train_index, test_index = index_list[i]
        test_indices = rand_perm[test_index]
        train_indices = rand_perm[train_index]
        #pick out data samples and labels corresponding to those indices
        train_data = train[train_indices,:] 
        test_data = train[test_indices,:]
        train_labels = trainlabels[train_indices]
        test_labels = trainlabels[test_indices]
        fold_accuracy = []
        for j in xrange(1,26,2):
            predicted_labels = knn(train_data, test_data, train_labels, j)
            compare_labels = predicted_labels - test_labels
            wrong_labels = filter(lambda a: a != 0, list(compare_labels))
            no_wrong_labels = len(wrong_labels)
            error = len(wrong_labels)/len(test_labels)
            accuracy = 1 - error
            fold_accuracy.append(accuracy)
        accuracy_lists.append(fold_accuracy)
        accuracy_matrix = np.array(accuracy_lists)
    print accuracy_matrix.shape
    
    k_accuracy_avg = np.mean(accuracy_matrix, axis = 0)
    k_best_index = np.argmax(k_accuracy_avg)

    k_values = np.arange(1,26,2)
    k_best = k_values[k_best_index]

    return k_best, accuracy_matrix
    pass
