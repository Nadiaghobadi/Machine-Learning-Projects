"""Basic code for assignment 1."""

import numpy as np
import pandas as pd
import scipy.stats as stats
import math

def load_unicef_data():
    """Loads Unicef data from CSV file.
    Retrieves a matrix of all rows and columns from Unicef child mortality
    dataset.

    Args:
      none
    Returns:
      Country names, feature names, and matrix of values as a tuple (countries, features, values).

      countries: vector of N country names
      features: vector of F feature names
      values: matrix N-by-F
    """
    fname = 'SOWC_combined_simple.csv'

    # Uses pandas to help with string-NaN-numeric data.
    data = pd.read_csv(fname, na_values='_')
    # Strip countries title from feature names.
    features = data.axes[1][1:]
    # Separate country names from feature values.
    countries = data.values[:,0]
    values = data.values[:,1:]
    # Convert to numpy matrix for real.
    values = np.asmatrix(values,dtype='float64')

    # Modify NaN values (missing values).
    mean_vals = np.nanmean(values, axis=0)
    inds = np.where(np.isnan(values))
    values[inds] = np.take(mean_vals, inds[1])
    return (countries, features, values)


def normalize_data(x):
    """Normalize each column of x to have mean 0 and variance 1.
    Note that a better way to normalize the data is to whiten the data (decorrelate dimensions).  This can be done using PCA.

    Args:
      input matrix of data to be normalized

    Returns:
      normalized version of input matrix with each column with 0 mean and unit variance

    """
    mvec = x.mean(0)
    stdvec = x.std(axis=0)

    return (x - mvec)/stdvec



def linear_regression(x, t, basis, degree):

    x_train = x
    t_train = t
    train_err = {}
    N_TRAIN = x_train.shape[0]
#    print feature_size
#    phi_train = np.matrix([[1]+[x_train.item(i, j)**d for d in range(1, degree+1) for j in range(feature_size)] for i in range(N_TRAIN)])
    phi_train = design_matrix(x_train, basis, degree)
    phi_cross = np.linalg.pinv(phi_train)
    #    print phi_cross.shape
    #    print t_train.shape
    w = phi_cross * t_train
#    print w.shape
    y_train = phi_train * w
    e = y_train - t_train
    e_train = math.sqrt((np.transpose(e) * e)/N_TRAIN)

    """Perform linear regression on a training set with specified regularizer lambda and basis

    Args:
      x is training inputs
      t is training targets
      reg_lambda is lambda to use for regularization tradeoff hyperparameter
      basis is string, name of basis to use
      degree is degree of polynomial to use (only for polynomial basis)

    Returns:
      w vector of learned coefficients
      train_err RMS error on training set
      """

    # TO DO:: Complete the design_matrix function.
    # e.g. phi = design_matrix(x,basis, degree)

    # TO DO:: Compute coefficients using phi matrix
#    w = None

    # Measure root mean squared error on training data.
#    train_err = None

    return (w, e_train)

def design_matrix(x, basis, degree=0):
    N = x.shape[0]
    feature_size = x.shape[1]

    """ Compute a design matrix Phi from given input datapoints and basis.
	Args:
      x matrix of input datapoints
      basis string name of basis

    Returns:
      phi design matrix
    """
    # TO DO:: Compute desing matrix for each of the basis functions
    if basis == 'polynomial':
        phi = np.matrix([[1]+[x.item(i, j)**d for d in range(1, degree+1) for j in range(feature_size)] for i in range(N)])
    elif basis == 'ReLU':
        phi = np.matrix([[1]+[max(0, 5000 - x.item(i, 0))] for i in range(N)])

    else:
        assert(False), 'Unknown basis %s' % basis

    return phi


def evaluate_regression(x, t, w, basis, degree):
    t_test = t
    x_test = x
    N_TEST = x_test.shape[0]
    feature_size = x_test.shape[1]
    print feature_size
    phi_test = design_matrix(x_test, basis, degree)
#        print phi_test.shape
#        print w.shape
    y_test = phi_test * w
    e_t = y_test - t_test

    e_test = math.sqrt((np.transpose(e_t) * e_t)/N_TEST)

    """Evaluate linear regression on a dataset.
	Args:
      x is evaluation (e.g. test) inputs
      w vector of learned coefficients
      basis is string, name of basis to use
      degree is degree of polynomial to use (only for polynomial basis)
      t is evaluation (e.g. test) targets

    Returns:
      t_est values of regression on inputs
      err RMS error on the input dataset
      """
  	# TO DO:: Compute t_est and err
#    t_est = None
#    err = None

    return (y_test, e_test)
