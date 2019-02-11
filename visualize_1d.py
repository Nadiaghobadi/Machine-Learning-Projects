#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt


(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,10:]
#x = a1.normalize_data(x)

N_TRAIN = 100;
# Select a single feature.
x_train = x[0:N_TRAIN]
t_train = targets[0:N_TRAIN]
x_test = x[N_TRAIN:,:]
t_test = targets[N_TRAIN:]

degree = 3
phi_train = np.matrix([[x_train.item(i, 0) ** d for d in range(0, degree+1)] for i in range(N_TRAIN)])
phi_cross = np.linalg.pinv(phi_train)
w = phi_cross * t_train
# Select a single feature.
#GNI per capital, feature 11
x_ev = x_train[:,0]

# Plot a curve showing learned function.
# Use linspace to get a set of samples on which to evaluate
x_sample = np.linspace(np.asscalar(min(x_ev)), np.asscalar(max(x_ev)), num=500)
phi_sample = np.matrix([[x_sample[i] ** d for d in range(0, degree+1)] for i in range(0, len(x_sample))])
y_sample = phi_sample * w
x_ev_test = x_test[:, 0]

# TO DO:: Put your regression estimate here in place of x_ev.
# Evaluate regression on the linspace samples.
#y_ev, _  = a1.evaluate_regression(x_test)
plt.figure(1)
plt.plot(x_sample,y_sample,'r.-')
plt.plot(x_ev,t_train,'go ')
plt.plot(x_ev_test,t_test,'bo')
plt.legend(['Polynomial','Traning points', 'Testing points'])
plt.title('A visualization of a regression estimate for GNI per capita')
plt.show()

#life expectancy, feature 12

phi_train = np.matrix([[x_train.item(i, 1) ** d for d in range(0, degree+1)] for i in range(N_TRAIN)])
phi_cross = np.linalg.pinv(phi_train)
w = phi_cross * t_train
# Select a single feature.

x_ev = x_train[:,1]
x_sample = np.linspace(np.asscalar(min(x_ev)), np.asscalar(max(x_ev)), num=500)
phi_sample = np.matrix([[x_sample[i] ** d for d in range(0, degree+1)] for i in range(0, len(x_sample))])
y_sample = phi_sample * w
x_ev_test = x_test[:, 1]
plt.figure(2)
plt.plot(x_sample,y_sample,'r.-')
plt.plot(x_ev,t_train,'bo')
plt.plot(x_ev_test, t_test, 'go')
plt.title('A visualization of a regression estimate for life expectancy')
plt.legend(['Polynomial','Traning points', 'Testing points'])
plt.show()

#literacy, feature 13
phi_train = np.matrix([[x_train.item(i, 2) ** d for d in range(0, degree+1)] for i in range(N_TRAIN)])
phi_cross = np.linalg.pinv(phi_train)
w = phi_cross * t_train
x_ev = x_train[:,2]
x_sample = np.linspace(np.asscalar(min(x_ev)), np.asscalar(max(x_ev)), num=500)
phi_sample = np.matrix([[x_sample[i] ** d for d in range(0, degree+1)] for i in range(0, len(x_sample))])
y_sample = phi_sample * w
x_ev_test = x_test[:, 2]
plt.figure(3)
plt.plot(x_sample,y_sample,'r.-')
plt.plot(x_ev,t_train,'bo')
plt.plot(x_ev_test, t_test, 'go')
plt.title('A visualization of a regression estimate for literacy')
plt.legend(['Polynomial','Traning points', 'Testing points'])
plt.show()
