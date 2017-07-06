from dca import dca
import numpy as np
import scipy as sp
from numpy import eye, ones, zeros, absolute
from scipy.linalg import orth, norm, qr
from scipy.spatial.distance import pdist, squareform
from numpy.matlib import repmat
import sys


# quick test file for DCA

# # generate fake data, where X1 and X2 share a latent variable Z
# Xs = []
# Z = np.random.randn(1,100);

# X1 = np.random.randn(3,100);
# X2 = np.random.randn(3,100);

# X1 = np.vstack((X1, Z));
# X2 = np.vstack((X2, Z));

# Xs.append(X1)
# Xs.append(X2)

# # generate a given distance matrix (Z)

# D1 = squareform(pdist(Z.transpose()));

# Ds = [];
# Ds.append(D1)


# # apply DCA

# # U, dcovs = dca(Xs, Ds = Ds, num_dca_dimensions = 3);  # DCA with full gradient descent
# U, dcovs = dca(Xs, Ds = Ds, num_dca_dimensions = 3, num_stoch_batch_samples=10);  # DCA stoch


# print dcovs
# print U[0]
# print U[1]


Xs = [];
Xs.append(np.random.randn(20,1000));
Xs.append(np.square(Xs[0][0:5,:]));
print Xs[0].shape
print Xs[1].shape
Ds = [];
Ds.append(squareform(pdist(Xs[0][0,:].reshape(1000,1) + Xs[1][0,:].reshape(1000,1))));

U, dcovs = dca(Xs, Ds = Ds, num_dca_dimensions = 5, percent_increase_criterion = 0.05);
print Xs[0].shape
print Xs[1].shape
U, dcovs = dca(Xs, Ds = Ds, num_dca_dimensions = 5, num_stoch_batch_samples = 100, 
              num_samples_to_compute_stepwise_dcov = 500, num_iters_foreach_dim = 20);

