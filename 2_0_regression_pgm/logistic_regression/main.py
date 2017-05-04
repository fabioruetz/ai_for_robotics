#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 12:13:01 2017

@author: sebastian
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


def logistic_function(w, x):
    # The given logistic function
    # Input: w  weight vector [dx1]
    # Input: x  data point of pixel xi [dx1]
    # TODO implement the logistic function
    L  = 1.0 / (1.0 + np.exp(- (w.T * x)))
    #assert(L.shape[0]<2)
    return L


def hessian_matrix(w, data_x, normalizer):
    # Calculates the hessian matrix which is the second derivative of the cost
    # function J with respect to w [dx1]
    # Input: w weigth vector [dx1]
    # Input data_x: data for Newtons method [[x1],[x2],[x3]...,[xn]], [nxd] xi[
    # n,1]
    # Output: H Hessian Matrix [dxd]
    H = np.zeros((data_x.shape[1], data_x.shape[1]))
    H = np.matrix(H)
    for x_i in range(data_x.shape[0]):
        x = np.matrix(data_x[x_i, :].reshape((data_x.shape[1], 1)))
        l = logistic_function(w,x)
        H = H + l[0,0]*(1 - l[0,0])*x*x.T

    # Normalizing H
    H = (H + np.identity(data_x.shape[1])*normalizer) / data_x.shape[1]
    return np.matrix(H)

def gradiant_J(w,data_x,y, normalizer):
    # Calculates the gradient of the cost function J with normalizer
    # input: w  weight vector [dx1]
    # input: data_x the train data, rows inout, columns features [mxd]
    # input: normalizer The normalizer, scalar
    # output: J gradient of cost function, J [dx1]
    J = np.zeros((data_x.shape[1], 1))
    J = np.matrix(J)
    for x_i in range(w.shape[0]):
        x = np.matrix(data_x[x_i, :].reshape((data_x.shape[1], 1)))
        l = logistic_function(w, x)
        J = J + (l[0,0] - y[x_i,0]) * x
    # Adding the regularizer term lambda*w
    J = ( J + normalizer * w ) / J.shape[0]
    return np.matrix(J)


def newtons_logosic_regression(w,x,y, iterations, normalizer):
    # Applies the newtons method for the logistic classification
    # input: w  weight vector [dx1]
    # input: x the train data, rows inout, columns features [mxd]
    # input: iterations How many NLR should be applied ( 5 -15 normally)
    # input: normalizer The normalizer, scalar
    # output: J gradient of cost function, J [dx1]

    # Do the gradient descent iter times with newtons method
    for iter in range(iterations):
        H = hessian_matrix(w, x, normalizer)
        H_inv = np.linalg.pinv(H)
        delta_J = gradiant_J(w, x, y, normalizer)
        w = w - H_inv*delta_J

    return w

# To make it easier the 24x24 pixels have been reshaped to a vector of 576 pixels. the value corrsponds to the greyscale intensity of the pixel
input_data = np.loadtxt("XtrainIMG.txt",delimiter=" ")# This is an  array
# that  has the features (all 576 pixel intensities) in the columns and all the available pictures in the rows
output_data = np.loadtxt("Ytrain.txt",delimiter=" ")  #  This is a vector that has the classification (1 for open eye 0 for closed eye) in the rows


n_samples = input_data.shape[0]
n_features = input_data.shape[1]

# Splitting data in train and validation sett
ratio_train_validate = 0.8
idx_switch = int(n_samples * ratio_train_validate)
training_input = np.matrix(input_data[:idx_switch, :])
training_output = np.matrix(output_data[:idx_switch][:,None])
validation_input = np.matrix(input_data[idx_switch:, :] )
validation_output = np.matrix( output_data[idx_switch:][:,None] )

# Initialize weight vector w with small non zero entries
w = np.matrix(np.ones((training_input.shape[1],1)))
w = 0.0001 * w

# Train w on the data
w = newtons_logosic_regression(w,training_input,training_output, 8,
                              normalizer=0.175)
# Validation
h = logistic_function(w,validation_input.T)
output = np.round(h).transpose()

error = np.abs(output-validation_output).sum()
print('wrong classification of ',(error/output.shape[0]*100),'% of the cases in the validation set')

# classify test data for evaluation
test_input = np.loadtxt("XtestIMG.txt",delimiter=" ")
h = logistic_function(w,test_input.T)
test_output = np.round(h.T)
np.savetxt('results.txt', test_output)
