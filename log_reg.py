#!/usr/local/EPD/bin/python
#Filename: log_reg.py

# Template procedure to perform logistic regression for a 2-class problem.
# Learn regression parameters with an unconstrained optimization routine.

# Overview:
# Take in csv file (without header)
# Split into training and test set
# Minimize cost function
# Compute performance metrics

# Code ported from logistic_regression_script.m, based on ml-class.org Ex.2

import numpy as np
import matplotlib.pyplot as plt
from numpy import mat, c_, r_, array

# Define functions



# Logistic regression script

# Input: feature columns followed by dependent class column at end
data = np.loadtxt('class_function_01.txt',delimiter=',')

# Percentage of data to use for training
train_perc = 0.7

# Separate input file into independent and dependent arrays
X = mat(data[:,:-1])
y = np.transpose(mat(data[:,-1]))

# Separate input file into training and test sets
test_rows = int(round(X.shape[0] * (1-train_perc))) #no. of rows in test set
X_test = X[:test_rows, :] #test set
y_test = y[:test_rows:] #test set

X = X[test_rows:,:] #training set
y = y[test_rows:] #training set

# Add intercept terms to sets

X = c_[np.ones(X.shape[0]), X]
X_test = c_[np.ones(X_test.shape[0]), X_test]














# EOF
