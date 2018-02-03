from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LogisticRegression

# input:  1) train data (without labels) in the form of a N by d numpy array, where N is the number of training data points and d is the number of dimensions
#         2) trainlabels: labels for training data in the form of a N by 1 numpy vector, where N is the number of training data points
#         3) test: test data (without labels) in the form of a M by d numpy array, where M is the number of test data points and d is the number of dimensions
#
# output: 1) vector (numpy array) consisting of the predicted classes for the test data
#         2) the beta parameter of the model as a (d+1) dimensional numpy array
#
# note: the labels should **not** be part of the train/test data matrices!
# note: The datamatrices are **NOT** already extended with a column of 1's - you should do it inside the function!.  
def logreg(train_data, train_labels, test_data):
    #based on code from logistic regression lecture
    #X = Xorig + column of 1s
    def logistic(input):
        out = np.exp(input)/(1+ np.exp(input))
        return out
    def logistic_insample(X, y, w):
        N, num_feat = X.shape
        E = 0
        for n in range(N):
            E = E + (1/N)*np.log(1/logistic(y[n]*np.dot(w,X[n,:])))
        return E
    def logistic_gradient(X, y, w):
        N, _ = X.shape
        g = 0*w

        for n in range(N):
            g = g + ((-1/N)*y[n]*X[n,:])*logistic(-y[n]*np.dot(w,X[n,:]))
        return g

    def gradient_descent(Xorig, y, t_max, grad_thr):
        #include column of 1s to train_data
        N, d = Xorig.shape
        X = np.c_[np.ones(N), Xorig]
        dplus1 = d + 1 #number of weights needed

        #initialise learning rate 
        eta = 0.1
        #initialise with random sample from normal distribution with weights at time step 0
        w = 0.1*np.random.randn(dplus1) #number of weights is equal to dimensions of training data + 1
        #compute logistic log likelihood
        E = logistic_insample(X, y, w)

        #initialise number of iterations and set boundary conditions
        t = 0
        convergence = 0

        #keep track of log likelihood values
        E_in = []

        #loop to iterate until the function converges?
        while convergence == 0:
            t = t + 1

            #compute the gradient of the log-likelihood wrt the current w
            g = logistic_gradient(X, y, w)

            #move in opposite direction of gradient to minimise the log likelihood
            v_t = -g

            #take a step in new direction
            eta_t = eta*np.linalg.norm(E_in)
            w_new = w + eta_t*v_t

            # Check for improvement
            # Compute in-sample error for new w
            E_t = logistic_insample(X, y, w_new)
            
            if E_t < E:
                w = w_new
                E = E_t
                E_in.append(E)
                eta *=1.05
            else:
                eta *= 0.95  

            #not sure about this part
            g_norm = np.linalg.norm(g)
            if g_norm < grad_thr:
                convergence = 1
            elif t > t_max:
                convergence = 1
            
            return w
        
    w = gradient_descent(train_data, train_labels, 10000000, 0.000)
    
    #use optimised weights to predict labels for test set
    def log_pred(Xorig, w):
        N, _ = Xorig.shape
        w_0 = w[0]
        w_new = w[1:]
        pred_classes = []
        for i in range(0,N):
            h_x = logistic(w[0] + np.dot((np.transpose(w_new)),Xorig[i,:]))
            if h_x > 0.5:
                pred_classes.append(1)
            else:
                pred_classes.append(0)
        pred_classes = np.array(pred_classes) 
        return pred_classes
    
    #using code from http://www.dummies.com/how-to/content/using-logistic-regression-in-python-for-data-scien.html
    logistic = LogisticRegression()
    logistic.fit(train_data, train_labels)
    pred_labels = logistic.predict(test_data)
        
    return pred_labels, w
    
    pass