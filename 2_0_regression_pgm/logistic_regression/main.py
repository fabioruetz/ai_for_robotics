#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 12:13:01 2017

@author: sebastian
"""
import numpy as np
import matplotlib.pyplot as plt

# The given logistic function
# Input: w  weight vector [dx1]
# Input: x  data point of pixel xi [dx1]
def logistic_function(w, x):
    # TODO implement the logistic function
    L  = 1.0 / (1 + np.exp(- w.transpose()* x))
    #assert(L.shape[0]<2)
    return L

# Calculates the hessian matrix which is the second derivative of the cost
# function J with respect to w [dx1]
# Input: w weigth vector [dx1]
# Input data_x: data for Newtons method [[x1],[x2],[x3]...,[xn]], [nxd] xi[
# n,1]
# Output: H Hessian Matrix [nxn]
def hessian_matrix(w, data_x, normalizer = False):
    H = np.zeros((data_x.shape[1], data_x.shape[1]))

    for x_i in range(data_x.shape[0]):
        # Make sure data has correct shape
        x = np.reshape(data_x[x_i,:],(data_x.shape[1],1))
        L = logistic_function(w,x)
        H += L[0,0]*(1-L[0,0]) * x*x.T

    if normalizer is True:
        H = H/data_x[0]

    # Check matrix dimension
    #assert(H.shape[0] |= data_x.shape[0])
    #assert (H.shape[1] not data_x.shape[0])
    return H

def gradiant_J(w,data_x,y, normalizer):

    J = np.zeros((data_x.shape[1], 1))
    for x_i in range(w.shape[0]):
        x = np.matrix(data_x[x_i, :].reshape((data_x.shape[1], 1)))
        J +=  x * (logistic_function(w,x) - y[x_i])
    if normalizer is True:
            J = J/data_x.shape[0] #  1/n n:data points
    else:
        pass

    return np.matrix(J)

def logistic_gradient_descent(w,x,y, iterations, normalizer = False):

    # Do the gradient descent iter times with newtons method
    for iter in range(iterations):
        H = hessian_matrix(w, x, normalizer)
        H_inv = np.linalg.pinv(H)
        delta_J = gradiant_J(w, x, y, normalizer)
        w = w - H_inv*delta_J

    return w

# To make it easier the 24x24 pixels have been reshaped to a vector of 576 pixels. the value corrsponds to the greyscale intensity of the pixel
input_data = np.loadtxt("XtrainIMG.txt",delimiter=" ")# This is an  array  that  has the features (all 576 pixel intensities) in the columns and all the available pictures in the rows
output_data = np.loadtxt("Ytrain.txt",delimiter=" ")  #  This is a vector that has the classification (1 for open eye 0 for closed eye) in the rows


n_samples = input_data.shape[0]
n_features = input_data.shape[1]


ratio_train_validate = 0.8
idx_switch = int(n_samples * ratio_train_validate)
training_input = np.matrix(input_data[:idx_switch, :])
training_output = np.matrix(output_data[:idx_switch][:,None])
validation_input = np.matrix(input_data[idx_switch:, :] )
validation_output = np.matrix( output_data[idx_switch:][:,None] )

# Initialize weight vector with random elements
w = np.matrix(np.random.rand(training_input.shape[1],1))


#TODO implement the iterative calculation of w
w = logistic_gradient_descent(w,training_input,training_output, 10, True)

#TODO2: modify the algorithm to account for regularization as well to improve the classifier

#validation
h = logistic_function(w,validation_input.T)
output = np.round(h).transpose()

error = np.abs(output-validation_output).sum()

print('wrong classification of ',(error/output.shape[0]*100),'% of the cases in the validation set')


# classify test data for evaluation
test_input = np.loadtxt("XtestIMG.txt",delimiter=" ")
h = logistic_function(w,test_input.T)
test_output = np.round(h)
np.savetxt('results.txt', test_output)
