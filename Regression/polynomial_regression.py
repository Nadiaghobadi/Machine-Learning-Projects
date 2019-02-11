#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt
import math

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
x = a1.normalize_data(x)

N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]
N_TEST = x_test.shape[0]
#1.a) country with highest child mortality rate in 1990
max_index = np.argmax(values[:,0])
print (countries[max_index])
print (values[max_index,0])

#1.b) Which country had the highest child mortality rate in 2011? What was the rate?

max_index = np.argmax(values[:,1])
print (countries[max_index])
print (values[max_index,1])

target_value = values[:,1]
input_features = values[:,7:40]

feature_size = x_train.shape[1]
phi=[]
#print x_train
train_err = {}
test_err = {}

# TO DO:: Complete the linear_regression and evaluate_regression functions of the assignment1.py
for degree in range(1, 7):
    (w, training_err) = a1.linear_regression(x_train, t_train, 'polynomial', degree)
    train_err[degree] = training_err
    (t_est, testing_err) = a1.evaluate_regression(x_test, t_test, w, 'polynomial', degree)
    test_err[degree] = testing_err

# Produce a plot of results.
plt.plot(train_err.keys(), train_err.values())
plt.plot(test_err.keys(), test_err.values())
plt.ylabel('RMS')
plt.legend(['Traning error','Test error'])
plt.title('Fit with polynomials, no regularization')
plt.xlabel('Polynomial degree')
plt.show()
