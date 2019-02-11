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

#print x_train

N_TRAIN_CROSS = 90
N_Validate = 10
degree = 2
lambd = [0, 0.01, 0.1, 1, 10, 100, 1000, 10000]
cross_err = []
for lam in lambd:
    cross_err_ave = 0
    for fold in range(10):
        x_train_cross = np.concatenate((x_train[0:fold*10], x_train[(fold+1)*10:]))
        t_train_cross = np.concatenate((t_train[0:fold*10], t_train[(fold+1)*10:]))
        x_test_cross = x_train[fold*10:(fold+1)*10]
        t_test_cross = t_train[fold*10:(fold+1)*10]

        phi_train = np.matrix([[1]+[x_train_cross.item(i, j)**d for d in range(1, degree+1) for j in range(x_train_cross.shape[1])] for i in range(90)])
        I = np.identity(phi_train.shape[1])
        phi_cross = np.linalg.inv(lam * I + phi_train.T * phi_train) * phi_train.T
        w = phi_cross * t_train_cross

        phi_validate = np.matrix([[1]+[x_test_cross.item(i, j)**d for d in range(1, degree+1) for j in range(x_test_cross.shape[1])] for i in range(x_test_cross.shape[0])])
        y_validate = phi_validate * w
        e = y_validate - t_test_cross
        x = e.T * e
        e_test = math.sqrt(x/N_Validate)
        cross_err_ave += e_test

    cross_err.append(cross_err_ave/10)

    cross_err_ave = 0

lambd = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
plt.semilogx(lambd, cross_err)
plt.ylabel('RMS')
plt.legend(['Cross validation error'])
plt.title('Polynomial regression with regularization')
plt.xlabel('Lambda')
plt.show()
