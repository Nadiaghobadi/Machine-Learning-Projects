#!/usr/bin/env python
# Run logistic regression training.

import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
import assignment2 as a2

# Maximum number of iterations.  Continue until this limit, or when error change is below tol.
max_iter = 500
tol = 0.00001

# Step size for gradient descent.
eta = [0.5, 0.3, 0.1, 0.05, 0.01]

# Load data.
data = np.genfromtxt('data.txt')
data = np.random.permutation(data)
# Data matrix, with column of ones at end.
X = data[:,0:3]
# Target values, 0 for class 1, 1 for class 2.
t = data[:,3]

plt.figure()
plt.ylabel('Negative log likelihood')
plt.title('Training logistic regression')
plt.xlabel('Epoch')

for et in eta:
    w = np.array([0.1, 0, 0])
    e_all = []

    for iter in range (0,max_iter):
        for i in range(int(len(X))):
            y = sps.expit(np.dot(X[i],w))
            grad_e = np.dot(t[i]-y ,X[i])
            w = w + np.dot(et, grad_e)

        e = -np.mean(np.multiply(t,np.log(y)) + np.multiply((1-t),np.log(1-y)))
        e_all.append(e)

        print 'epoch {0:d}, negative log-likelihood {1:.4f}, w={2}'.format(iter, e, w.T)
        if iter>0:
            if np.absolute(e-e_all[iter-1]) < tol:
                break
    plt.plot(e_all, label = et)

plt.legend()
plt.show()
