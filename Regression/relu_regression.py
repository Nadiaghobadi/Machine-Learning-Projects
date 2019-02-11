#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt
import math

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,10:]
#x = a1.normalize_data(x)

N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]
N_TEST = x_test.shape[0]

target_value = values[:,1]
#input_features = values[:,7:40]

feature_size = x_train.shape[1]
phi=[]
#print x_train

degree = 0
# TO DO:: Complete the linear_regression and evaluate_regression functions of the assignment1.py
(w, training_err) = a1.linear_regression(x_train, t_train, 'ReLU', degree)
train_err= training_err
(t_est, testing_err) = a1.evaluate_regression(x_test, t_test, w, 'ReLU', degree)
test_err = testing_err
print ("training error is: "+ str(training_err))
print ("testing error is: " + str(testing_err))
x_ev = x_train[:,0]
x_sample = np.linspace(np.asscalar(min(x_ev)), np.asscalar(max(x_ev)), num=500)
phi_sample = np.matrix([[1]+[max(0, (5000 - x_sample[i]))] for i in range(len(x_sample))])
y_sample = phi_sample * w
#print w.shape
#print phi_sample.shape
#print y_sample.shape

plt.plot(x_sample,y_sample,'r.-')
plt.plot(x_ev,t_train,'bo')
plt.title('A visualization of a ReLU regression estimate for GNI over training points')
plt.legend(['Polynomial','Traning points'])

plt.show()

# Produce a plot of results.
#plt.plot(train_err.keys(), train_err.values())
#plt.plot(test_err.keys(), test_err.values())
#plt.ylabel('RMS')
#plt.legend(['Traning error','Test error'])
#plt.title('Fit with polynomials, no regularization')
#plt.xlabel('Polynomial degree')
#plt.show()
