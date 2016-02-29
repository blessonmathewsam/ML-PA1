import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import random
import math
import winsound
import time

def nnPredict(w1,w2,data):

    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % label: a column vector of predicted labels"""

    labels = np.array([])

    # set the number of nodes in input unit (not including bias unit)
    n_input = data.shape[1];

    # set the number of nodes in hidden unit (not including bias unit)
    n_hidden = w1.shape[0];

    # set the number of nodes in output unit
    n_class = w2.shape[0];

    bias = np.ones((data.shape[0],1))
    data = np.append(data,bias, axis =1)

    for p in range(0,data.shape[0]):
        #print "Minimize:"+str(p)
        z = np.empty(n_hidden+1)
        for j in range(0,n_hidden):
            z[j] = sigmoid(np.dot(w1[j,:],data[p,:]))
        z[n_hidden] = 1
        o = np.empty([n_class,1])
        max = -1
        o_class = -1
        for l in range(0,n_class):
            o[l] = sigmoid(np.dot(w2[l,:],z))
            if(o[l]>max):
                max = o[l]
                o_class = l
        labels = np.hstack((labels,np.array([o_class])))
    return labels

"""**************Prediction Script Starts here********************************"""

#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset

print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
