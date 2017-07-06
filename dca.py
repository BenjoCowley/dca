import numpy as np
import scipy as sp
from numpy import eye, ones, zeros, absolute
from scipy.linalg import orth, norm, qr
from scipy.spatial.distance import pdist, squareform
from numpy.matlib import repmat
import sys


def dca(Xs, Ds = [], num_iters_per_dataset = 1, num_iters_foreach_dim = 30, 
        percent_increase_criterion = 0.01, num_dca_dimensions = [], 
        num_stoch_batch_samples = 0, num_samples_to_compute_stepwise_dcov = 1000,
        u_0s = []):
#  U, dcovs = dca(Xs, ...)
#
#  DESCRIPTION:
#   DCA identifies linear projections in sets of variables that are related to one
#       another, linearly or nonlinearly.  DCA returns a loading matrix U[iset] for each
#       set of variables, as well as the distance covariances dcovs for each dimension.
#
#  INPUTS:
#   Xs: (1 x M list), data, M datasets where Xs[iset] is (num_variables x num_samples) ndarray
#                   Note: Each set can have a different number of variables,
#                   but must have the same number of samples.
#   Ds (optional): (1 x N list), distance matrices of N datasets for which dimensions
#                   are *not* identified, but are related to dimensions of Xs.
#                   Ds[iset] is (num_samples x num_samples), and must have the same
#                   number of samples as those in Xs
#
#   Additional (optional) arguments:
#       num_iters_per_dataset: (1 x 1), number of optimization iterations for each dataset; default: 1
#       num_iters_foreach_dim: (1 x 1), number of optimization iterations for each dimension; default: 30
#       percent_increase_criterion (1 x 1 scalar, between 0 and 1), if objective value of next iteration
#                               does not surpass a fraction of the previous objective value, stop optimizaiton;
#                               default: 0.01
#       num_dca_dimensions: (1 x 1), number of dimensions to optimize; default: num_variables
#       num_stoch_batch_samples: (1 x 1), number of samples in minibatch for stochastic gradient
#                       descent; default: 0
#                       Note: A nonzero value for this option triggers stochastic gradient descent.
#                       Use for large datasets (e.g., num_vars > 100, num_samples > 5000). A good
#                       default: 100 samples.
#       num_samples_to_compute_stepwise_dcov: (1 x 1), for stochastic gradient descent, number 
#                       of samples used to compute dcovs (for visualization purposes); default: 1000 
#
#  OUTPUTS:
#   U: (1 x M list), orthonormal loading matrices for M datasets in Xs.
#               U[iset] is (num_variables x num_dca_dimensions).
#   dcovs: (1 x num_dca_dimensions list), distance covariances for each dimension identified by DCA
#
#  EXAMPLE:
#   import numpy as np
#   from scipy.spatial.distance import pdist, squareform
#   Xs = [];
#   Xs.append(np.random.randn(20,1000));
#   Xs.append(np.square(Xs[0][1:5,:]));
#   Ds = [];
#   Ds.append(squareform(pdist(X[0][1,:].transpose() + Xs[1][1,:].transpose())));
#   U, dcovs = dca(Xs, Ds = Ds, num_dca_dimensions = 5, percent_increase_criterion = 0.05);
#   U, dcovs = dca(Xs, Ds = Ds, num_dca_dimensions = 5, num_stoch_batch_samples = 100, 
#                   num_samples_to_compute_stepwise_dcov = 500, num_iters_foreach_dim = 20);
#   

            
    ### pre-processing
        check_input(Xs, Ds, num_dca_dimensions); # checks user input, outputs warnings/errors

        results = preprocessing(Xs, Ds, num_dca_dimensions, num_stoch_batch_samples);

        results['num_samples_to_compute_stepwise_dcov'] = num_samples_to_compute_stepwise_dcov;
        results['percent_increase_criterion'] = percent_increase_criterion;
        results['num_iters_foreach_dim'] = num_iters_foreach_dim;
        results['num_iters_per_dataset'] = num_iters_per_dataset;

        num_dca_dimensions = results['num_dca_dimensions'];
        num_datasets = results['num_datasets'];
        num_samples = results['num_samples'];
        R_given = results['R_given'];
        dcovs = results['dcovs'][0];
        U_orth = results['U_orth']; 
        U = results['U'];   # list, dca dimensions for each dataset
        Xs = results['Xs'];
        Xs_orig = results['Xs_orig'];
        Xij_orig = results['Xij_orig'];
        Xij = results['Xij'];
        col_indices = results['col_indices'];
        D_given = results['D_given'];
        



    ### optimization
        for idim in range(num_dca_dimensions):

            print "dca dimension {}".format(idim+1);

            u, momented_gradf, R, total_dcov, total_dcov_old, stoch_learning_rate, results = initialization(Xs, u_0s, results);  
                    # compute re-centered distance matrices based on u and update stoch grad parameters


            itotal_dcov = 1;  # keeps track of number of iterations after a run across all datasets

            while (check_if_dcov_increases(total_dcov, total_dcov_old, itotal_dcov, results)):
                # if dcov does not increase by a certain percentage
                # or if we reached the number of iterations, stop

                print "   step {}: dcov = {}".format(itotal_dcov, total_dcov);

                r = np.random.permutation(num_datasets);  # randomize the order of datasets being optimized

                if (num_stoch_batch_samples == 0):
                    ### Projected gradient descent, uses all samples

                    sys.stdout.write('        sets:');
                    for iset in range(0,num_datasets):
                        s = " {} ".format(r[iset]+1);
                        sys.stdout.write(s);
                        Rtemp = R[:];
                        del Rtemp[r[iset]];
                        R_combined = get_recentered_combined(Rtemp, R_given); # get combined recentered distance matrix (summed)

                        # perform optimization for one dataset
                        u[r[iset]] = dca_one(Xs[r[iset]], Xij[r[iset]],
                                                      R_combined, u[r[iset]], col_indices, results);

                        R[r[iset]] = get_recentered_matrix(u[r[iset]], Xs[r[iset]]);

                    total_dcov_old = total_dcov;
                    total_dcov = get_total_dcov(R, D_given);

                else:
                    ### Stochastic projected gradient descent, mini-batch

                    random_sample_indices = np.random.permutation(num_samples);
                    batch_indices = range(0,num_samples, num_stoch_batch_samples);

                    print "     batches:";

                    for ibatch in range(0,len(batch_indices)-1):  # ignore last set of samples since randomized

                        sys.stdout.write('.');

                        window = np.arange(batch_indices[ibatch], batch_indices[ibatch+1]);
                        sample_indices = random_sample_indices[window];

                        R = [];
                        for iset in range(num_datasets):
                            R.append(get_recentered_matrix(u[iset], Xs[iset][:,sample_indices]));

                        for iset in range(num_datasets):
                            Rtemp = R[:];
                            del Rtemp[r[iset]];
                            R_giventemp = R_given[:, sample_indices];
                            R_giventemp = R_giventemp[sample_indices,:];
                            R_combined_sampled = get_recentered_combined(Rtemp, R_giventemp);
                            Xij_sampled = get_Xij_randomlysampled(Xs[r[iset]][:,sample_indices]);

                            # perform optimization for one dataset
                            u[r[iset]], momented_gradf[r[iset]] = dca_one_stoch(Xs[r[iset]][:,sample_indices], Xij_sampled, R_combined_sampled,
                                                     u[r[iset]], stoch_learning_rate, momented_gradf[r[iset]], col_indices);

                            R[r[iset]] = get_recentered_matrix(u[r[iset]], Xs[r[iset]][:,sample_indices]);

                    total_dcov_old = total_dcov;
                    total_dcov = get_total_dcov_randomlysampled(u, Xs, D_given, results);

                    stoch_learning_rate = 0.9 * stoch_learning_rate; # other forms of learning rates possible, like 1/sqrt(itotal_dcov)


                itotal_dcov = itotal_dcov + 1;
                print "";
            sys.stdout.write('\n');
            dcovs[idim] = total_dcov;

        ### Project data onto orthogonal subspace of U

            # ensure that the u are normalized
            for iset in range(num_datasets):
                u[iset] = u[iset] / norm(u[iset]);

            # project identified dca dimension into original space
            for iset in range(num_datasets):
                v = np.dot(U_orth[iset], u[iset]);   # U_orth{iset} * u{iset}
                U[iset][:,idim] =  v[:,0];


            # project data onto null space of newly found dca dimensions
            if (idim + 1 != num_dca_dimensions):
                for iset in range(num_datasets):

                    Ubases = np.concatenate((U[iset][:,0:idim+1], 
                                    np.random.randn(U[iset].shape[0], U[iset].shape[0] - idim - 1)), 1);

                    Q, chugs = qr(Ubases);
                    U_orth[iset] = Q[:,(idim+1):Q.shape[1]];

                    Xs[iset] = np.dot(U_orth[iset].transpose(), Xs_orig[iset]);
                                # U_orth{iset}' * Xs_orig{iset}

                    if (num_stoch_batch_samples == 0):
                        Xij[iset] = np.dot(U_orth[iset].transpose(), Xij_orig[iset]);
                                # only used for full gradient descent
                                #   U_orth{iset}' * Xij_orig{iset}

        # sort distance covariances and basis vectors
        #       since they may not be in order due to noise
        sorted_indices = np.fliplr(np.argsort(dcovs).reshape((1,num_dca_dimensions)))
        dcovs = np.fliplr(np.sort(dcovs).reshape((1,num_dca_dimensions)));

        for iset in range(num_datasets):
            U[iset] = U[iset][:,sorted_indices[0,:]];

        return U, dcovs;







#################################
##    DCA HELPER FUNCTIONS  ##
#################################

def check_input(Xs, Ds, num_dca_dimensions):
# make sure user inputs correct formats

    ### check Xs
        if (not isinstance(Xs, list)): # check if Xs is a list
            raise NameError('Xs must be a list of arrays, where Xs[0] is (num_vars x num_samples).');

        if (len(Xs) == 0):
            raise NameError('Xs was empty. Xs must be a list of arrays, where Xs[0] is (num_vars x num_samples).');

        for array in Xs:
            if (not isinstance(array, np.ndarray)):
                raise NameError('Xs must be a list of arrays, where Xs[0] is (num_vars x num_samples).');

        num_datasets = len(Xs)

        num_samples = Xs[0].shape[1]
        for array in Xs:
            if (num_samples != array.shape[1] or num_samples == 0):
                raise NameError('Dataset(s) in Xs do not contain the same number of samples. Xs[iset] (num_vars x num_samples), where num_samples is the same for each dataset (but num_vars can be different).');

        isnan_found = False
        for iset in range(0, num_datasets):
            isnan_found = isnan_found or np.isnan(Xs[iset]).any();
        if (isnan_found == True):
            raise NameError('Dataset(s) in Xs contain NaNs. Remove samples with NaNs.');


    ### check D

        if (not isinstance(Ds, list)):
            raise NameError('Ds can either be empty (Ds = []) or a list.')

        if (len(Ds) > 0):
            num_samples_Ds = Ds[0].shape[1];
            for array in Ds:
                if (num_samples != array.shape[0] or num_samples != array.shape[1] or num_samples == 0):
                    raise NameError('Dataset(s) in Ds do not have the same number of samples. Ds[iset] (num_samples x num_samples) are distance matrices, where num_samples is the same for each dataset.')

            isnan_found = False
            for iset in range(0, len(Ds)):
                isnan_found = isnan_found or np.isnan(Ds[iset]).any();
            if (isnan_found == True):
                raise NameError('Dataset(s) in Ds contain NaNs. Remove samples with NaNs.')

            isnegative_found = False
            for iset in range(0,len(Ds)):
                isnegative_found = isnegative_found or np.any(Ds[iset] < 0)
            if (isnegative_found == True):
                raise NameError('Dataset(s) in Ds contain negative values.  Ds[iset] is a distance matrix and should have nonnegative values.');


    ### check that X and D have more than just one dataset combined
        if (len(Xs) + len(Ds) <= 1):
            raise NameError('Not enough datasets in Xs and Ds. The number of datasets (including given distance matrices) should be at least two.');


    ### check that X and D have the same number of samples
        if (len(Ds) > 0 and num_samples != num_samples_Ds):
            raise NameError('Dataset(s) in Xs do not have the same number of samples as those in Ds. They should be the same.');


    ### check num_dca_dimensions
        min_num_of_dimensions = np.inf;
        for array in Xs:
            num_vars = array.shape[0]
            if (num_vars < min_num_of_dimensions):
                min_num_of_dimensions = num_vars;
        if (num_dca_dimensions != [] and num_dca_dimensions > min_num_of_dimensions):
            s = 'num_dca_dimensions must be less than or equal to ' + str(min_num_of_dimensions) + ', the minimum number of variables across datasets.';
            raise NameError(s)



def preprocessing(Xs, Ds, num_dca_dimensions, num_stoch_batch_samples):
# compute any fixed variables before optimization, and initialize any needed book-keeping quantities

    ### declare numbers of quantities
        Xs_orig = Xs;  # Xs_orig will remain the original Xs, since Xs will change during optimization
        Xs = Xs_orig[:];  

        num_datasets = len(Xs);
        num_samples = Xs[0].shape[1];

    ### check how many dca dimensions there should be
        if (num_dca_dimensions == []):
            # check for minimum number of dimensions across datasets (else user already input the number of dca dimensions)
            min_num_of_dimensions = np.inf;
            for array in Xs:
                num_vars = array.shape[0]
                if (num_vars < min_num_of_dimensions):
                    min_num_of_dimensions = num_vars;
            num_dca_dimensions = min_num_of_dimensions;



    ### compute the combined recentered distance matrices for given matrices
        D_given = Ds[:];
        R_given = np.zeros((num_samples, num_samples));
        if (len(D_given) > 0):
            for iset in range(len(D_given)):
                H = eye(num_samples) - 1.0 / num_samples * ones((num_samples, num_samples));  # recenters distance matrix
                D_given[iset] = np.dot(np.dot(H, D_given[iset]), H);  # H * D_given[iset] * H
                R_given = R_given + D_given[iset];

            R_given = R_given / len(D_given);



    ### prepare indices for column indices when subtracting off distance matrix means

        col_indices = np.empty((1,0));
        if (num_stoch_batch_samples == 0):   # full gradient descent
            for icol in range(num_samples):
                col_indices = np.append(col_indices, range(icol,num_samples ** 2, num_samples));
        else:
            # stochastic gradient descent
            for icol in range(num_stoch_batch_samples):
                col_indices = np.append(col_indices, range(icol, num_stoch_batch_samples ** 2, num_stoch_batch_samples));
        col_indices = np.int_(col_indices);


    ### instantiate parameters

        dcovs = np.zeros((1,num_dca_dimensions));  # vector, dcovs for each dimension
        U = [];
        for iset in range(num_datasets):
            num_vars = Xs[iset].shape[0];
            U.append(np.zeros((num_vars, num_dca_dimensions)));

        U_orth = [];
        for iset in range(num_datasets):
            num_vars = Xs[iset].shape[0];    # number of variables could be different for each set
            U_orth.append(np.eye(num_vars));  # keeps track of orthogonal space of U
                


    ###  compute Xij (num_variables x num_samples^2) for each dataset, where Xij = X_i - X_j
        Xij_orig = [];
        Xij = [];
        if (num_stoch_batch_samples == 0):    # compute only for full gradient descent
            for iset in range(0,num_datasets):
                # compute all combinations of differences between samples of Xs
                #    this is some fancy footwork in order to compute quickly...
                num_vars = Xs[iset].shape[0];
                X = Xs[iset][..., None];
                X = np.transpose(X, (0,2,1));
                X = Xs[iset][..., None] - X;
                X = -np.reshape(X, (num_vars, -1), order='F')
                Xij.append(X);
            Xij_orig = Xij[:];

        return {'num_datasets':num_datasets, 'Xs':Xs, 'Xs_orig':Xs_orig, 'num_dca_dimensions':num_dca_dimensions, 
                'D_given':D_given, 'U':U, 'U_orth':U_orth, 'dcovs':dcovs, 'Xij_orig':Xij_orig, 'Xij':Xij,
                'num_stoch_batch_samples':num_stoch_batch_samples, 'R_given':R_given, 'dcovs':dcovs,
                'col_indices':col_indices, 'num_samples':num_samples}






def initialization(Xs, u_0s, results):
    # results is a dictionary from preprocessing
    # initialize U, U_orth, and dcovs
    # U_orth keeps track of the orthogonal space of U


    ### for first dim, initialize u with either user input or randomly
        num_datasets = results['num_datasets'];
        u = [];
        for iset in range(num_datasets):
            num_vars = Xs[iset].shape[0];
            if (u_0s != [] and u_0s[iset].shape[0] == num_vars):  # if user input initialized weights for first dim
                u.append(u_0s[iset][:,1]);
            else:
                u.append(orth(np.random.randn(num_vars, 1)));

    ### get initial recentered matrices for each dataset based on u
        R = [];
        if (results['num_stoch_batch_samples'] == 0): # only for full gradient descent
            for iset in range(num_datasets):
                R.append(get_recentered_matrix(u[iset], Xs[iset]));

            total_dcov = get_total_dcov(R,results['D_given']);
            total_dcov_old = total_dcov * 0.5; # set old value to half, so it'll pass threshold


    ### stochastic gradient descent initialization
        momented_gradf = [];
        stoch_learning_rate = 1;  # initial learning rate for SGD
        if (results['num_stoch_batch_samples'] > 0):
            for iset in range(num_datasets):
                momented_gradf.append(np.zeros(u[iset].shape));

            total_dcov = get_total_dcov_randomlysampled(u, Xs, results['D_given'], results);
            total_dcov_old = total_dcov * 0.5;

        return u, momented_gradf, R, total_dcov, total_dcov_old, stoch_learning_rate, results;



def get_recentered_matrix(u, X):
    # computes the recentered distance matrix for each dataset
    # u: (num_variables x 1), weight vector
    # X: (num_variables x num_samples), one dataset

    # compute distance matrix of X projected onto u
        D = squareform(pdist(np.transpose(np.dot(np.transpose(u), X))));
                # squareform(pdist((u' * X)')

    # now re-center it
        num_samples = X.shape[1];
        H = np.eye(num_samples) - 1.0 / num_samples * np.ones((num_samples, num_samples));
        R = np.dot(np.dot(H, D), H);  # R = H * D * H;

        return R;


def get_total_dcov(R, D_given):
    # compute the total distance covariance across all datasets
    # R: (1 x num_datasets), re-centered matrices
    # D_given: (1 x num_given_datasets), combined re-centered matrix for given distance matrices

        R = R + D_given;   # concatenates both lists

        Rtotal = 0;
        T = R[0].shape[0]; # number of samples
        for iset in range(len(R)):
            for jset in range(iset+1,len(R)):
                Rtotal = Rtotal + np.sqrt(np.dot(1.0/(T ** 2), np.dot(R[iset].flatten('F'), R[jset].flatten('F'))));
                    # Rtotal = Rtotal + sqrt(1/T^2 * R{iset}(:)' * R{jset}(:));


        total_dcov = Rtotal / ((len(R) - 1) * len(R)/np.float64(2));

        return total_dcov;



def get_total_dcov_randomlysampled(u, Xs, D_given, results):
    # computes dcov for a random subsample (for stochastic gradient descent)

        num_samples = Xs[0].shape[1];
        r = np.random.permutation(num_samples);

        sample_indices = r[0:np.minimum(len(r), results['num_samples_to_compute_stepwise_dcov'])-1];

        T = len(sample_indices); # T = number of samples in a batch

        R = [];
        for iset in range(len(Xs)):
            R.append(get_recentered_matrix(u[iset], Xs[iset][:,sample_indices]));

        for iset in range(len(D_given)):
            Dtemp = D_given[iset][sample_indices,:];
            Dtemp = Dtemp[:,sample_indices];
            R.append(Dtemp);
                # this is an approximation, we would really need to re-compute the re-centered distance matrix for each D_given

        Rtotal = 0;
        for iset in range(len(R)):
            for jset in range(iset+1,len(R)):
                Rtotal = Rtotal + np.sqrt(np.dot(1.0/T**2, np.dot(R[iset].flatten('F'), R[jset].flatten('F'))));
                    # Rtotal = Rtotal + sqrt(1/T^2 * R{iset}(:)' * R{jset}(:));  

        total_dcov = Rtotal / ((len(R)-1) * len(R) / 2);

        return total_dcov;




def check_if_dcov_increases(total_dcov, total_dcov_old, itotal_dcov, results):
    # returns true if increase in dcov is greater than the percent threshold
    #   or if the number of iterations is less than iteration constraint
    #   else returns false

        if (results['num_stoch_batch_samples'] == 0):  # full gradient descent
            percent_increase = absolute(total_dcov - total_dcov_old)/absolute(total_dcov_old);

            if (total_dcov - total_dcov_old < 0.0): # if value goes down, stop
                result = False;
            elif (percent_increase >= results['percent_increase_criterion'] and
                    itotal_dcov <= results['num_iters_foreach_dim']):
                result = True;
            else:
                result = False;
        else:  # stochastic gradient descent...just check number of iterations
            if (itotal_dcov <= results['num_iters_foreach_dim']):
                result = True;
            else:
                result = False;

        return result;




def get_recentered_combined(R, R_given):
    # compute the combined matrix of all re-centered distance matrices
    # returns a matrix, where each element is a pointwise-sum of all R
    # and R_given (remember, R_given is already re-centered)

        # initialize R_combined as a zero matrix
        if (R != []):
            R_combined = np.zeros((R[0].shape[0], R[0].shape[0]));
        else:
            R_combined = np.zeros((R_given.shape[0]));

        # iterate and add through R
        for iset in range(len(R)):
            R_combined = R_combined + R[iset] / len(R);

        # incorporate R_given, if given
        if (np.all(R_given == 0) == False):  # if R-given ~= 0, then D_given exists
            R_combined = (R_combined + R_given)/2.0;

        return R_combined;



def get_Xij_randomlysampled(X):
    # computes Xij for a subsample (for stochastic gradient descent)

    # compute all combinations of differences between samples of X
        num_vars = X.shape[0];
        Xreshaped = X[..., None];
        Xreshaped = np.transpose(Xreshaped, (0,2,1));
        X = X[..., None] - Xreshaped;
        Xij_sampled = -np.reshape(X, (num_vars, -1), order='F');

        return Xij_sampled;




######################################
##  DCA_ONE - FULL GRADIENT DESCENT ##
######################################

def dca_one(X, Xij, R_combined, u_0, column_indices, results):
  # performs distance covariance analysis for one dataset and one given re-centered distance matrix
  #     uses projected gradient descent
  #
  # X: (N x T), data in which we want to find the N x 1 dca dimension, where
  #             N is the number of variables, and T is the number of samples
  # R_combined: (T x T), combined re-centered distance matrix of the other sets of variables
  # u_0: (N x 1), initial guess for the dca dimension
  # results: (1 x 1), dictionary which contains user constraints, such as the number of iterations
  #
  #  returns:
  #     u: (N x 1), the dimension of greatest distance covariance between D_X and R_combined


    ### pre-processing
        N = X.shape[0];
        T = X.shape[1];

        if (np.sum(np.var(X,axis=1)) < 1e-10): # check if X has little variability left
            u = np.random.randn(N,1);
            u = u / norm(u);
            return u;

        u = u_0;  # set u to be the initial guess


    ### optimization
        for istep in range(results['num_iters_per_dataset']):  # stop when num iters have been reached

          # compute gradient descent parameters
            D_uXij = get_D_uXij(u, X);  # get distance matrix of current u
            f_val = get_f(D_uXij, R_combined, T);   # compute current function value and gradf for backtracking
            gradf = get_gradf(u, D_uXij, Xij, R_combined, N, T, column_indices);  # compute gradient of current solution

          # Backtracking line search
            # first check large intervals for t (so backtracking loop doesn't take forever)
            for candidate_power in range(1,10):
                sys.stdout.write('.');
                t = 10.0**(-candidate_power);
                if (backtrack_check(u, f_val, gradf, t, R_combined, T, X) == False):
                    break;


            # find more nuanced t
            while (backtrack_check(u, f_val, gradf, t, R_combined, T, X) and t > 1e-9):
                t = 0.7 * t;

          # Perform projected grad descent
            u_unnorm = u - t * gradf;   # gradient descent step

            norm_u = norm(u_unnorm);  # project u_unnorm onto the L2 unit ball
            if (norm_u > 1):
                u = u_unnorm / norm_u;
            else:
                u = u_unnorm;  # allow solution to exist inside unit ball (for dca_one)

        return u;




##############################
## DCA_ONE HELPER FUNCTIONS ##
##############################

def get_D_uXij(u, X):
    # compute distance matrix of (X projected onto u)

    D = squareform(pdist(np.transpose(np.dot(np.transpose(u), X))));
        # D = squareform(pdist((u' * X)'))

    return D;



def get_f(D_uXij, R_combined, T):
    # compute objective function

    H = eye(T) - 1.0/T * ones(T);
    A = np.dot(H, np.dot(D_uXij, H)); #  A = H * D_uXij * H, recenters distance matrix
    f = -np.dot(R_combined.flatten().transpose(), A.flatten());  # trying to minimize, so flip the sign!

    return f;


def get_gradf(u, D_uXij, Xij, R_combined, N, T, column_indices):
    # computes the gradient for dca_one...note there are some tricky matrix operations in here!

        ### weight Xij
            D_uXij[D_uXij == 0] = 1e-8;  # add 1e-8 to avoid division by zero...
                                         #    other constant values do not matter (since divided out)
            # project Xij onto u
            XijT_u = np.dot(np.transpose(Xij), u);

            # weight Xij by XijT_u ./ D_uXij(:)'
            Xij_weighted = Xij * np.transpose(XijT_u.flatten('F') / D_uXij.flatten('F'));


        ### subtract row, column, and matrix means  (magic...)
            Xij_temp = np.reshape(Xij_weighted, (N, T, T), order='F');
            Xij_temp = np.mean(Xij_temp, axis=1);
            Xij_col_means = repmat(Xij_temp, 1, T);
            Xij_row_means = Xij_col_means[:,column_indices];
            Xij_matrix_mean = np.mean(Xij_weighted,axis=1);

            Xij_weighted = Xij_weighted - Xij_row_means - Xij_col_means + Xij_matrix_mean.reshape(Xij_weighted.shape[0],1);

        ### linearly combine with R_combined
            gradf = - np.dot(Xij_weighted, R_combined.flatten('F'));  # flip sign because we are minimizing negative dcov

            return gradf.reshape(gradf.shape[0],1);





def backtrack_check(u, f_next, gradf, t, R_combined, T, X):
    # check lecture 8 of ryan tibshirani's opti class, spring 2014

    Gt = get_Gt(u, gradf, t);

    D_uXij_t = get_D_uXij(u - t * Gt, X);
    status = get_f(D_uXij_t, R_combined, T) > f_next - t * np.dot(gradf.transpose(), Gt) + t/2.0 * np.dot(Gt.transpose(), Gt);

    return status;



def get_Gt(u, gradf, t):
    # vector used for backtracking check with projected gradient descent

    u_n = u - t * gradf;
    norm_u_n = norm(u_n);

    if (norm_u_n > 1.0): # project to L2 unit ball
        u_norm = u_n / norm_u_n;
    else:
        u_norm = u_n;

    Gt = 1.0/t * (u - u_norm);

    return Gt






###################################################
##  DCA_ONE STOCH  - STOCHASTIC GRADIENT DESCENT ##
###################################################

def dca_one_stoch(X, Xij, R_combined, u_0, learning_rate, old_momented_grad_f, column_indices):
    # performs distance covariance analysis for one dataset and one given re-centered distance matrix
    #       uses stochastic projected gradient descent
    #
    #  INPUT:
    #   X: (N x T), data in which we want to find the (N x 1) dca dimension, where
    #               N is the number of variables, and T is the number of samples
    #   R_combined: (T x T), combined re-centered distance matrix of the other sets of variables
    #   u_0: (N x 1), initial guess for the dca dimension
    #   learning_rate: (1 x 1), current learning rate for stochastic gradient descent
    #   old_momented_grad_f: (N x 1), previous gradient direction (used for momentum)
    #   column_indices: (1 x T^2), used to subtract out the column means of the gradient
    #
    #  OUTPUT:
    #   u: (N x 1), the dimension of greatest distance covariance between D_X and R_combined
    #   momented_gradf: (N x 1), needed to keep momentum terms during alternating optimization


    ### Pre-processing
        N = X.shape[0];
        T = X.shape[1];

        if (np.sum(np.var(X,axis=1)) < 1e-10): # check if X has little variability left
            u = np.random.randn(N,1);
            u = u / norm(u);
            momented_gradf = zeros(u.shape);
            return u, momented_gradf;

        u = u_0;  # set u to be the initial guess
        

    ### Optimization

        # Compute gradient descent parameters
        momentum_weight = 1.0 - learning_rate;  # momentum term convex combination of learning rate
        D_uXij = get_D_uXij(u, X);  # get distance matrix of current u

        # perform projected gradient descent with momentum
        gradf = get_gradf(u, D_uXij, Xij, R_combined, N, T, column_indices);
                        # works better than Nesterov accelerated gradient

        momented_gradf = learning_rate * gradf + momentum_weight * old_momented_grad_f;
        u_unnorm = u - momented_gradf;  # gradient descent step


        norm_u = norm(u_unnorm);  # project u_unnorm to the L2 unit ball
        if (norm_u > 1.0):
            u = u_unnorm / norm_u;
        else:
            u = u_unnorm;  # allow solution to exist in L2 ball (for dca_one_stoch)

        return u, momented_gradf;

