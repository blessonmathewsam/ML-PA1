import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import random
import math
import winsound
import time

def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W



def sigmoid(z):

    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return  1/(1+np.exp(-z))



def stackMatrix(type,mat):
    matrix = mat.get(type + str(0))
    label = np.empty(matrix.shape[0])
    label.fill(0)
    for i in range(1,10):
        nmatrix = mat.get(type + str(i))
        matrix = np.vstack((matrix,nmatrix))
        nlabel = np.empty(nmatrix.shape[0])
        nlabel.fill(i)
        label = np.hstack((label,nlabel))
    return matrix, label



def removeFeatures(matrix,dims):
    j = 0
    while(j<matrix.shape[1]):
        if(j in dims):
            a = matrix[:,[j]]
            j += 1
            break
        j += 1
    while(j<matrix.shape[1]):
        if(j in dims):
            b = matrix[:,[j]]
            a = np.concatenate((a, b), axis=1)
        j += 1
    return a



def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""

    mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary

    #Pick a reasonable size for validation data


    #Your code here
    trainData, trainLabel = stackMatrix('train',mat)
    testData, testLabel = stackMatrix('test',mat)
    dims = []
    for j in range(0,trainData.shape[1]):
        change = False
        val = trainData[0][j]
        for i in range(1,trainData.shape[0]):
            if(trainData[i][j] != val):
                change = True
                break
        if(change):
            dims.append(j)

    trainData = removeFeatures(trainData,dims)
    testData = removeFeatures(testData,dims)

    trainMax = trainData.max()
    testMax = testData.max()

    trainData = trainData/float(trainMax)
    testData = testData/float(testMax)

    A = trainData
    a = range(A.shape[0])
    aperm = np.random.permutation(a)
    validationData = A[aperm[0:10000],:]
    trainData = A[aperm[10000:],:]

    A = trainLabel
    validationLabel = A[aperm[0:10000]]
    trainLabel = A[aperm[10000:]]

    return trainData, trainLabel, validationData, validationLabel, testData, testLabel


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log
    %   likelihood error function with regularization) given the parameters
    %   of Neural Networks, thetraining data, their corresponding training
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    #Your code here

    final_grad_w1 = np.zeros([n_hidden,n_input+1])
    final_grad_w2 = np.zeros([n_class,n_hidden+1])

    bias = np.ones((training_data.shape[0],1))
    training_data = np.append(training_data,bias, axis =1)

    #1-of-K Encoding
    encoded_label = np.empty([training_label.shape[0],10])
    for i in range(0,training_label.shape[0]):
        encoded_label[i][training_label[i]] = 1

    print "Minimize:"
    for p in range(0,training_data.shape[0]):
        grad_w1 = np.zeros([n_hidden,n_input+1])
        grad_w2 = np.zeros([n_class,n_hidden+1])
        squared_loss = 0
        z = np.empty(n_hidden+1)

        for j in range(0,n_hidden):
            z[j] = sigmoid(np.dot(w1[j,:],training_data[p,:]))
        z[n_hidden] = 1
        o = np.empty([n_class,1])
        delta = np.empty(n_class)
        for l in range(0,n_class):
            o[l] = sigmoid(np.dot(w2[l,:],z))
            delta[l] = (encoded_label[p][l]-o[l])*(1-o[l])*o[l]
            squared_loss += (encoded_label[p][l]-o[l])*(encoded_label[p][l]-o[l])
            grad_w2[l] = np.dot(-1*delta[l],z)
        for j in range(0,n_hidden):
            dw = np.dot(delta,w2[:,j])
            x = -1.0*(1-z[j])*z[j]*dw
            grad_w1[j] = np.dot(x,training_data[p,:])

        obj_val += squared_loss/2
        final_grad_w1 += grad_w1
        final_grad_w2 += grad_w2

    final_grad_w1 = np.dot(1/float(training_data.shape[0]),final_grad_w1)
    final_grad_w2 = np.dot(1/float(training_data.shape[0]),final_grad_w2)


    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    #obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.concatenate((final_grad_w1.flatten(), final_grad_w2.flatten()),0)
    obj_val /= training_data.shape[0]
    print obj_val

    return (obj_val,obj_grad)

"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();

print train_label.shape
#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1];

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 10;

# set the number of nodes in output unit
n_class = 10;

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0;


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

np.save('w1.npy',w1)
np.save('w2.npy',w2)

i = 0
while(i<10):
    winsound.Beep(400,500)
    time.sleep(0.25)
    i += 1
