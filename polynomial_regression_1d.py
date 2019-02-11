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

target_value = values[:,1]
input_features = values[:,7:40]
feature_size = x_train.shape[1]
phi=[]
#print x_train
train_err = {}
test_err = {}
degree = 3
for j in range(0, 8):

    phi_train = np.matrix([[1]+[x_train.item(i, j)**d for d in range(1, degree+1)] for i in range(N_TRAIN)])

    phi_cross = np.linalg.pinv(phi_train)
    print phi_cross.shape
    print t_train.shape

    w = phi_cross * t_train

    y_train = phi_train * w
    e = y_train - t_train
    e_train = math.sqrt((np.transpose(e) * e)/N_TRAIN)
    train_err[j] = e_train

    phi_test = np.matrix([[1]+[x_test.item(i, j)**d for d in range(1, degree+1)] for i in range(N_TEST)])
    y_test = phi_test * w
    e_t = y_test - t_test
    e_test = math.sqrt((np.transpose(e_t) * e_t)/N_TEST)
    test_err[j] = e_test

bar_width = 0.3
index = np.arange(8)+8


plt.bar(index, train_err.values(), width = bar_width, color = 'green')
plt.bar(index+bar_width, test_err.values(), width = bar_width, color = 'blue')
# Produce a plot of results.
#plt.plot(train_err.keys(), train_err.values())
#plt.plot(test_err.keys(), test_err.values())
plt.ylabel('RMS')
plt.legend(['Traning error','Test error'])
plt.title('Fit with polynomials, no regularization')
plt.xlabel('Feature ID')
plt.show()
