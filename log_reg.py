#!/usr/local/EPD/bin/python
#Filename: log_reg.py

# Template procedure to perform logistic regression for a 2-class problem.
# Learn regression parameters with SciPy's optimize package.

# Overview:
# Take in csv file (without header)
# Split into training and test set
# Minimize cost function
# Compute performance metrics

# Code ported from logistic_regression_script.m, based on ml-class.org Ex.2

import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy import mat, c_, r_, array, e, reshape
from scipy import optimize as op

# Define functions

def sigmoid(z):
	g = 1./(1 + e**(-z))
	return g

def costFunction(theta,X,y): #computes cost given predicted and actual values
	m = X.shape[0] #number of training examples
	
	J = (1/m) * (-np.transpose(y).dot(np.log(sigmoid(X.dot(theta)))) - np.transpose(1-y).dot(np.log(1-sigmoid(X.dot(theta)))))
	
	grad = np.transpose((1/m)*(sigmoid(X.dot(theta)) - y).dot(X))
	return J,grad

def misclassError(y,y_hat,thresh):
	#y = actual value
	#y_hat = predicted value
	#thresh = probability threshold for 2-class variable
	
	m = X.shape[0] #number of training examples
	#testError = (1/m) * sum(double(y_hat>=thresh
	testError = 0
	return testError


# Logistic regression script

# Input: feature columns followed by dependent class column at end
data = np.loadtxt('class_function_01.txt',delimiter=',')

train_perc = 0.77# Percentage of data to use for training
thresh = 0.5 # Threshold for classifying hypothesis output

# Separate input file into independent and dependent arrays
X = array(data[:,:-1])
y = array(data[:,-1])
y = reshape(y,(len(y),1))

# Separate input file into training and test sets
test_rows = int(round(X.shape[0] * (1-train_perc))) #no. of rows in test set
X_test = X[:test_rows, :] #test set
y_test = y[:test_rows:] #test set

X = X[test_rows:,:] #training set
y = y[test_rows:] #training set

# Add intercept terms to sets
X = c_[np.ones(X.shape[0]), X]
X_test = c_[np.ones(X_test.shape[0]), X_test]

# Initialize fitting parameters
initial_theta = array(np.zeros((X.shape[1],1)))

# Set options for optimization
options = {'full_output':True, 'maxiter':400}

sys.exit()
# Run fmin
theta, cost, _, _, _, = op.fmin(lambda t: costFunction(t,X,y), initial_theta, **options)






# EOF
