function [U, dcovs] = dca(Xs, varargin)
% [U, dcovs] = dca(Xs, Ds, 'Option', value, ...)
%
% DESCRIPTION:
%   DCA identifies linear projections in sets of variables that are related to
%       one another, linearly or nonlinearly.  DCA returns a loading matrix
%       U{iset} for each set of variables, as well as the distance covariances
%       dcovs for each dimension.
%
% INPUTS:
%   Xs: (1 x M cell), data, M datasets where Xs{iset} is (num_variables x num_samples)
%           Note: Each set can have a different number of variables,
%           but must have the same number of (paired) samples.
%   Ds (optional): (1 x N cell), distance matrices of N datasets for which dimensions
%           are *not* identified, but are related to dimensions of Xs
%           Ds{jset} is num_samples x num_samples  (and must have the same
%           number of samples as those in Xs
%   
%   Additional (optional) arguments:
%       'num_dca_dimensions': (1 x 1 scalar), number of dimensions to
%           optimize; default: num_variables
%       'num_iters_per_dataset': (1 x 1 scalar), number of optimization
%           iterations for each dataset; default: 1
%       'num_iters_foreach_dim': (1 x 1 scalar), number of optimization
%           iterations for each dimension; default: 30
%       'percent_increase_criterion': (1 x 1 scalar between 0 and 1),
%           if objective value of next iteration does not surpass
%           a fraction of the previous object value, stop optimization;
%           default: 0.01
%       'num_stoch_batch_samples': (1 x 1 scalar), number of samples
%           in minibatch for stochastic gradient descent; default: 0
%           Note: A nonzero value for this option triggers stochastic 
%           gradient descent. Use for large datasets (e.g., num_vars > 100, 
%           num_samples > 5000). A good default: 100 samples.
%       'num_samples_to_compute_stepwise_dcov': (1 x 1 scalar), for
%           stochastic gradient descent, number of samples used to compute
%           dcovs (for visualization purposes); default: 1000
%
% OUTPUTS:
%   U: (1 x M cell), orthonormal loading matrices for M datasets in Xs.  
%           U{iset} is (num_variables x num_dca_dimensions)
%   dcovs: (1 x num_dca_dimensions vector), distance covariances for each
%           dimension identified by DCA
%
% EXAMPLE:
%   Xs{1} = randn(20,1000);
%   Xs{2} = Xs{1}(1:5,:).^2;
%   Ds{1} = squareform(pdist(Xs{1}(1,:)' + Xs{2}(1,:)'));
%   [U, dcovs] = dca(Xs, Ds, 'num_dca_dimensions', 5);
%   % plot first DCA dimension in Xs{1} vs Xs{2}
%   x1 = U{1}(:,1)' * Xs{1}
%   x2 = U{2}(:,1)' * Xs{2}
%   plot(x1, x2, '.k'); xlabel('DCA dim 1 for Xs{1}'); ylabel('DCA dim 1 for Xs{2}')
%
% EXAMPLE for SGD (large datasets):
%   [U, dcovs] = dca(Xs, Ds, 'num_dca_dimensions', 5, 'num_stoch_batch_samples', 100, ...
%                   'num_samples_to_compute_stepwise_dcov', 500, 'num_iters_foreach_dim', 20);
%
%
% Reference: 
%   BR Cowley, JD Semedo, A Zandvakili, A Kohn, MA Smith, BM Yu. "Distance
%       covariance analysis." In AISTATS, pp. 242-251, 2017.
%
% Author: Benjamin R. Cowley, March 2017, bcowley@cs.cmu.edu
%   updated Jan 2021 for Matlab 2020b


    % X_orig = [];
    % Xij_orig = [];
    % num_datasets = 0;
    % num_samples = 0;
    % num_dca_dims = 0;
    % R_given = [];
    % D_given = [];
    % col_indices = [];

    %%% pre-processing
        global X_orig Xij_orig num_datasets num_samples num_dca_dims R_given D_given col_indices
            % currently used as global. Future update will move these to a pass-by-reference struct.

        p = parse_input(Xs, varargin);   % allows user to input name-value pairs

        check_input(p);  % checks user input, outputing warnings

        preprocessing(p);  % initialize parameters
   
    %%% optimization
        for idim = 1:num_dca_dims

            fprintf('dca dimension %d\n', idim);

            initialization(p);  % compute re-centered distance matrices based on u
                        % and update stoch grad parameters

            itotal_dcov = 1; % keeps track of number of iterations after a run across all datasets

            while (check_if_dcov_increases(p, total_dcov, total_dcov_old, itotal_dcov)) 
                            % if dcov does not increase by a certain percentage
                            % or if we reached the number of iterations, stop

                fprintf('  step %d: dcov = %f\n', itotal_dcov, total_dcov);
                
                r = randperm(num_datasets);  % randomize the order of datasets being optimized

                if (p.Results.num_stoch_batch_samples == 0)
                    %%% PROJECTED GRADIENT DESCENT, ALL SAMPLES

                    fprintf('    sets:');
                    for iset = 1:num_datasets

                        fprintf(' %d ', r(iset));

                        R_combined = get_recentered_combined(R((1:end)~=r(iset)), R_given);  % get combined recentered distance matrix (summed)
                                
                        % perform optimization for one dataset
                        u{r(iset)} = dca_one(X{r(iset)}, Xij{r(iset)}, R_combined, u{r(iset)}, col_indices, p);
                        
                        R{r(iset)} = get_recentered_matrix(u{r(iset)}, X{r(iset)});
                    end
                    
                    total_dcov_old = total_dcov;
                    total_dcov = get_total_dcov(R, D_given);
                else
                    %%% STOCHASTIC PROJECTED GRADIENT DESCENT, MINI-BATCH
                    
                    random_sample_indices = randperm(num_samples);
                    batch_indices = 1:p.Results.num_stoch_batch_samples:num_samples;
                    fprintf('    batches:');
                    
                    for ibatch = 1:length(batch_indices)-1 % ignore last set of samples since randomized

                        fprintf('.');
                        window = batch_indices(ibatch):batch_indices(ibatch+1)-1;
                        sample_indices = random_sample_indices(window);

                        R = cell(1,num_datasets);
                        for iset = 1:num_datasets
                            R{iset} = get_recentered_matrix(u{iset}, X{iset}(:,sample_indices));
                        end

                        for iset = 1:num_datasets
                            R_combined_sampled = get_recentered_combined(R((1:end) ~= r(iset)), R_given(sample_indices, sample_indices));
                            Xij_sampled = get_Xij_randomlysampled(X{r(iset)}(:,sample_indices));

                            % perform optimization for one dataset
                            [u{r(iset)}, momented_gradf{r(iset)}] = dca_one_stoch(X{r(iset)}(:,sample_indices), ...
                                Xij_sampled, R_combined_sampled, u{r(iset)}, ...
                                stoch_learning_rate, momented_gradf{r(iset)}, col_indices);

                            R{r(iset)} = get_recentered_matrix(u{r(iset)}, X{r(iset)}(:,sample_indices));
                        end
                    end

                    total_dcov_old = total_dcov;
                    total_dcov = get_total_dcov_randomlysampled(u, X, D_given, p);
                    
                    stoch_learning_rate = 0.9 * stoch_learning_rate;  % other forms of learning rates possible, like 1/sqrt(itotal_dcov)
                end

                itotal_dcov = itotal_dcov + 1;
                fprintf('\n');

            end

            dcovs(idim) = total_dcov;
            
            %%% PROJECT DATA ONTO ORTHOGONAL SUBSPACE OF U
                
                % ensure that the u are normalized
                for iset = 1:num_datasets
                    u{iset} = u{iset} ./ norm(u{iset});
                end

                % project identified dca dimension into original space
                for iset = 1:num_datasets
                    U{iset}(:,idim) = U_orth{iset} * u{iset};
                end

                % project data onto null space of newly found dca dimensions 
                if (idim ~= num_dca_dims)
                    for iset = 1:num_datasets
                        [Q,R] = qr([U{iset}(:,1:idim) ...
                                randn(size(U{iset},1), size(U{iset},1)-idim)]);
                        U_orth{iset} = Q(:,(idim+1):end);

                        X{iset} = U_orth{iset}' * X_orig{iset};
                        
                        if (p.Results.num_stoch_batch_samples == 0)
                            Xij{iset} = U_orth{iset}' * Xij_orig{iset};  % only used for full gradient descent
                        end
                    end
                end
        end

        % sort distance covariances/patterns, 
        %       since it may not be in order if large noise
        [dcovs, sorted_indices] = sort(dcovs, 'descend');

        for iset = 1:num_datasets
            U{iset} = U{iset}(:,sorted_indices);
        end
        
        
        
        
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%
    % NESTED HELPER FUNCTIONS
    %%%%%%%%%%%%%%%%%%%%%%%%%
    
    function p = parse_input(X, vargs)
        % parses input, and extracts name-value pairs
        
        p = inputParser;  % creates parser object
        
        default_D = [];  % distance matrices
        default_num_iters_per_dataset = 1;
        default_num_iters_foreach_dim = 30;
        default_percent_increase_criterion = 0.01;  % stops when objective function increases fewer than 1% of current value
        default_num_dca_dimensions = []; % number of dca dimensions to identify
        default_u_0s = [];  % how to intialize the dimensions before optimization
        default_num_stoch_batch_samples = 0;
        default_num_samples_to_compute_stepwise_dcov = 1000;
        
        addRequired(p, 'X');
        addOptional(p, 'Ds', default_D);
        addParamValue(p, 'num_iters_per_dataset', default_num_iters_per_dataset);
        addParamValue(p, 'num_iters_foreach_dim', default_num_iters_foreach_dim);
        addParamValue(p, 'percent_increase_criterion', default_percent_increase_criterion);
        addParamValue(p, 'num_dca_dimensions', default_num_dca_dimensions);
        addParamValue(p, 'num_stoch_batch_samples', default_num_stoch_batch_samples);
        addParamValue(p, 'num_samples_to_compute_stepwise_dcov', default_num_samples_to_compute_stepwise_dcov);
        addParamValue(p, 'u_0s', default_u_0s);
        
        % NOTE: addParamValue should be changed to addParameter...
        %      addParamValue is for older matlab versions
        
        parse(p,X,vargs{:});  % parses input to get optional parameters
            % to get input, use p.Results.X, etc.
    end

    function check_input(p)
        % make sure user inputs correct formats
        
        %%% check X
            if (~iscell(p.Results.X) || size(p.Results.X,1) > 1 && size(p.Results.X,2) > 1) % check if X is a cell vector  
                error('Xs (1 x num_datasets) should be a cell array, where Xs{iset} is (num_variables x num_datapoints)');
            end
            
            num_datasets = length(p.Results.X);
            
            [num_vars, num_samples] = cellfun(@size, p.Results.X);
            if (length(unique(num_samples)) ~= 1)  % there should only be one sample size
                error('Dataset(s) in Xs do not contain the same number of samples. Xs{iset} (num_variables x num_samples), where num_samples is the same for each dataset (but num_variables can be different).');
            end
            num_samples = size(p.Results.X{1},2);
            
            isnan_found = false;
            for iset = 1:num_datasets
                isnan_found = isnan_found | any(any(isnan(p.Results.X{iset})));
            end
            if (isnan_found == true)
                error('Dataset(s) in Xs contain NaNs. Remove samples with NaNs.');
            end
            
        %%% check Ds
            if (~isempty(p.Results.Ds) && (~iscell(p.Results.Ds) || size(p.Results.Ds, 1) > 1 && size(p.Results.Ds, 2) > 1))
                erorr('Ds should either be empty (Ds = []) or a cell vector');
            end
            
            if (length(p.Results.Ds) > 0)
                [num_samples1, num_samples2] = cellfun(@size, p.Results.Ds);
                if (length(unique([num_samples1, num_samples2])) ~= 1)
                    error('Dataset(s) in Ds do not contain the same number of samples. Ds{iset} (num_samples x num_samples) are distance matrices, where num_samples is the same for each dataset.');
                end

                isnan_found = false;
                for iset = 1:length(p.Results.Ds)
                    isnan_found = isnan_found | any(any(isnan(p.Results.Ds{iset})));
                end
                if (isnan_found == true)
                    error('Dataset(s) in Ds contain NaNs. Remove samples with NaNs.');
                end

                isneg_found = false;
                for iset = 1:length(p.Results.Ds)
                    isneg_found = isneg_found | any(any(p.Results.Ds{iset} < 0));
                end
                if (isneg_found == true)
                    error('Dataset(s) in Ds contain negative values. Ds{iset} is a distance matrix with nonnegative values.');
                end
            end
            
        %%% check that X and D have more than just one dataset combined
            if (length(p.Results.X) + length(p.Results.Ds) <= 1)
                error('Not enough datasets in Xs and Ds.  The number of datasets (including given distance matrices) should be at least two.');
            end
            
        %%% check that X and D have the same number of samples
            if (~isempty(p.Results.Ds) && num_samples ~= num_samples1)
                error('Dataset(s) in Xs do not have the same number of samples as those in Ds.  They should be the same.');
            end
            
        %%% check num_dca_dimensions
            if (p.Results.num_dca_dimensions > min(num_vars))
                error(sprintf('"num_dca_dimensions" must be less than or equal to %d, the minimum number of variables across datasets.', min(num_vars)));
            end
    end


    function preprocessing(p)
        % - compute any fixed variables before optimization
        % - initialize any needed quantities
    
        X = p.Results.X;  % X will change as we optimize each dim
        X_orig = X;   % X_orig will remain the original X
        
        %%% check how many dca dimensions there should be
        %   for minimum number of dimensions across datasets + user input
            num_dims_foreach_set = [];
            for iset = 1:num_datasets
                num_dims_foreach_set = [num_dims_foreach_set size(X{iset},1)];
            end
            num_dca_dims = min(num_dims_foreach_set);

            if (~isempty(p.Results.num_dca_dimensions))
                num_dca_dims = p.Results.num_dca_dimensions;
            end        

        %%% compute the combined recentered matrices for given distance matrices
            R_given = zeros(num_samples);
            D_given = p.Results.Ds;
            if (~isempty(D_given))
                for iset = 1:length(D_given)
                    H = eye(size(D_given{iset})) - 1 / size(D_given{iset},1) * ones(size(D_given{iset}));
                    D_given{iset} = H * D_given{iset} * H;  % recenter D
                    R_given = R_given + D_given{iset};
                end

                R_given = R_given / length(D_given);
            end
        
        %%% prepare indices for column indices when subtracting off distance matrix means
            if (p.Results.num_stoch_batch_samples == 0)  % full gradient descent
                col_indices = [];
                for icol = 1:num_samples
                    col_indices = [col_indices icol:num_samples:num_samples^2];
                end
            else     % stochastic gradient descent
                col_indices = [];
                for icol = 1:p.Results.num_stoch_batch_samples
                    col_indices = [col_indices icol:p.Results.num_stoch_batch_samples:p.Results.num_stoch_batch_samples^2];
                end
            end

        %%% initialize parameters
            U = cell(1,num_datasets); % cell vector, dca dimensions for each dataset
            dcovs = zeros(1,num_dca_dims); % vector, dcovs for each dimension
            for iset = 1:num_datasets
                U_orth{iset} = eye(size(X{iset},1));  % keeps track of the orthogonal space of u
            end
            
        %%% compute Xij (num_neurons x num_samples^2) for each dataset, where Xij = X_i - X_j
            if (p.Results.num_stoch_batch_samples == 0)  % only for full grad descent
                Xij = [];
                for iset = 1:num_datasets
                    % compute all combinations of differences between samples of X
                    Xij{iset} = bsxfun(@minus, X{iset}, permute(X{iset}, [1 3 2]));
                    Xij{iset} = reshape(-Xij{iset}, size(X{iset},1), []);
                end
                Xij_orig = Xij;
            end

    end


    function initialization(p)
        % initialize U, U_orth, and dcovs
        % U_orth keeps track of the null space of U
        
        %%% for first dim, initialize u with either user input or randomly
        for iset = 1:num_datasets
            if (~isempty(p.Results.u_0s) && size(p.Results.u_0s{iset},1) == size(X{iset},1))  % if user input initialized weights for first dim
                u{iset} = p.Results.u_0s{iset}(:,1);
            else
                u{iset} = orth(randn(size(X{iset},1), 1));
            end
        end

        %%% get initial recentered matrices for each dataset based on u
            if (p.Results.num_stoch_batch_samples == 0)  % only for full grad descent
                R = cell(1,num_datasets);
                for iset = 1:num_datasets
                    R{iset} = get_recentered_matrix(u{iset}, X{iset});
                end
                
                total_dcov = get_total_dcov(R,D_given);
                total_dcov_old = total_dcov * 0.5;  % set old value to half, so it'll pass threshold
            end
        
        % stochastic gradient descent initialization
            if (p.Results.num_stoch_batch_samples > 0)
                stoch_learning_rate = 1;  % initial learning rate for SGD
                momented_gradf = cell(1,num_datasets);
                for iset = 1:num_datasets
                    momented_gradf{iset} = zeros(size(u{iset}));
                end

                total_dcov = get_total_dcov_randomlysampled(u, X, D_given, p);
                total_dcov_old = total_dcov * 0.5;
            end
    end
end





%%%%%%%%%%%%%%%%%%%%%%%
%  NON-NESTED FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%


function R = get_recentered_matrix(u, X)
    % computes the recentered distance matrix for each dataset
    % u: (num_variables x 1),  weight vector
    % X: (num_variables x num_datapoints), one dataset

    % compute distance matrix of X projected onto u
        D = squareform(pdist((u' * X)'));
        
    % now recenter it
        H = eye(size(D)) - 1/size(D,1) * ones(size(D));
        R = H * D * H;  % recenters distance matrix
end


        
function total_dcov = get_total_dcov(R,D_given)
    % compute the total distance covariance across all datasets
    % R: (1 x num_datasets), re-centered matrices
    % D_given: (1 x num_given_datasets), combined re-centered matrix for given distance matrices

    R = [R D_given];
    
    Rtotal = 0;
    T = size(R{1},1);
    for iset = 1:length(R)
        for jset = (iset+1):length(R)
            Rtotal = Rtotal + sqrt(1/T^2 * R{iset}(:)' * R{jset}(:));  
        end
    end

    total_dcov = Rtotal / ((length(R)-1)*length(R)/2);
    
end


function total_dcov = get_total_dcov_randomlysampled(u, X, D_given, p)
 % computes dcov for a random subsample (for stochastic gradient descent)
 

    r = randperm(size(X{1},2));
    sample_indices = r(1:min(length(r), p.Results.num_samples_to_compute_stepwise_dcov));
    
    T = length(sample_indices);  % T = number of subsamples
    
    R = cell(1,length(X) + length(D_given));
    for iset = 1:length(X)
        R{iset} = get_recentered_matrix(u{iset}, X{iset}(:,sample_indices));
    end

    for iset = 1:length(D_given)
        R{iset + length(X)} = D_given{iset}(sample_indices, sample_indices);  % this is an approximation, we would really need to re-compute the re-centered distance matrix for each D_given
    end
    
    Rtotal = 0;
    for iset = 1:length(R)
        for jset = (iset+1):length(R)
            Rtotal = Rtotal + sqrt(1/T^2 * R{iset}(:)' * R{jset}(:));  
        end
    end

    total_dcov = Rtotal / ((length(R)-1)*length(R)/2);
end


function result = check_if_dcov_increases(p, total_dcov, total_dcov_old, itotal_dcov)
    % returns true if increase in dcov is greater than the percent threshold
    %   or if the number of iterations is less than iteration constraint
    % else returns false

    if (p.Results.num_stoch_batch_samples == 0) % full gradient descent
        percent_increase = abs(total_dcov - total_dcov_old)/abs(total_dcov_old);

        if (total_dcov - total_dcov_old < 0)  % if value goes down, stop
            result = false;
        elseif (percent_increase >= p.Results.percent_increase_criterion && ...
                    itotal_dcov <= p.Results.num_iters_foreach_dim)
            result = true;
        else
            result = false;
        end
    else  % stochastic gradient descent...just check number of iterations
        if (itotal_dcov <= p.Results.num_iters_foreach_dim)
            result = true;
        else
            result = false;
        end
    end
end


function R_combined = get_recentered_combined(R, R_given)
    % compute the combined matrix of all re-centered distance matrices
    % returns a matrix, where each element is a pointwise-sum of all R
    % and R_given (remember, R_given is already re-centered)
    
    % initialize R_combined as a zero matrix
    if (~isempty(R))  
        R_combined = zeros(size(R{1}));  
    else
        R_combined = zeros(size(R_given));  % 
    end
    
    % iterate and add through R
    for iset = 1:length(R)
        R_combined = R_combined + R{iset}/length(R);
    end
    
    % incorporate R_given, if given
    if (~all(all(R_given==0))) % if R_given ~= 0, then D_given exists
        R_combined = (R_combined + R_given)/2;
    end

end


function Xij_sampled = get_Xij_randomlysampled(X)
% computes Xij for a subsample   (for stochastic gradient descent)

    % compute all combinations of differences between samples of X
    Xij_sampled = bsxfun(@minus, X, permute(X, [1 3 2]));
    Xij_sampled = reshape(-Xij_sampled, size(X,1), []);
end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  DCA ONE - FULL GRADIENT DESCENT  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function u = dca_one(X, Xij, R_combined, u_0, column_indices, p)
    % performs distance covariance analysis for one dataset and one given re-centered distance matrix
    %       uses projected gradient descent
    % 
    %   X: (N x T), data in which we want to find the N x 1 dca dimension, where
    %       N is the number of variables, and T is the number of samples
    %   R_combined: (T x T), combined re-centered distance matrix of the other sets of variables
    %   u_0: (N x 1), initial guess for the dca dimension
    %   p: (1 x 1), inputParser object which contains user constraints, such
    %               as number of iterations
    %
    %  returns:
    %   u: (N x 1), the dimension of greatest distance covariance between D_X and R_combined

    %%% PRE-PROCESSING
        N = size(X,1);  % number of neurons
        T = size(X,2);  % number of timepoints
        
        if (sum(var(X')) < 1e-10) % X has little variability left
            u = randn(N,1);
            u = u / norm(u);
            return;
        end

        u = u_0;    % set u to be initial guess
        
    %%% OPTIMIZATION  
        for istep = 1:p.Results.num_iters_per_dataset  % stop when num iters have been reached

            % COMPUTE GRAD DESCENT PARAMETERS
                D_uXij = get_D_uXij(u);  % get distance matrix of current u
                f_val = get_f(D_uXij);     % compute current function value and gradf for backtracking
                gradf = get_gradf(u, D_uXij);  % compute gradient of current solution

                t = 1;  % backtracking step size

            % BACKTRACKING LINE SEARCH
                % first check large intervals for t (so backtracking loop doesn't take forever)
                for candidate_power = 1:9
                    fprintf('.');
                    if (~backtrack_check(u, f_val, gradf, 10^-candidate_power))
                        break;
                    else
                        t = 10^-candidate_power;
                    end
                end

                % find more nuanced t
                while (backtrack_check(u, f_val, gradf, t) && t > 10^-9)
                    t = 0.7 * t;
                    fprintf('.')
                end 

            % PERFORM PROJECTED GRAD DESCENT
                u_unnorm = u - t * gradf; % gradient descent step

                norm_u = norm(u_unnorm); % project u_unnorm to the L2 unit ball
                if (norm_u > 1)
                    u = u_unnorm / norm_u;
                else
                    u = u_unnorm;   % allow solution to exist inside unit ball (for dca_one)
                end
        end

        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % NESTED DCA_ONE HELPER FUNCTIONS %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function D = get_D_uXij(u) 
        % compute distance matrix of (X projected onto u)
        
        D = squareform(pdist((u' * X)'));
    end

    function f = get_f(D_uXij)
        % compute objective function
        
        H = eye(T) - 1/T * ones(T);
        A = H * D_uXij * H;  % recenters distance matrix
        f = -R_combined(:)' * A(:);  % trying to minimize, so flip the sign!
    end

    function gradf = get_gradf(u, D_uXij)
        % computes the gradient for dca_one...note there are some tricky matrix operations in here!
        
        %%% weight Xij
            D_uXij(D_uXij == 0) = 1e-8; % add 1e-8 to avoid division by zero...other values
                                        % do not seem to matter
            % project Xij onto u
            XijT_u = Xij' * u;
        
            % weight Xij by XijT_u ./ D_uXij(:)'
            Xij_weighted = bsxfun(@times, Xij, (XijT_u ./ D_uXij(:))');

        %%% subtract row, column, and matrix means
            Xij_row_means = blockproc(Xij_weighted, [N, T], @get_row_means);
            Xij_col_means = Xij_row_means(:,column_indices);
            Xij_matrix_mean = mean(Xij_weighted,2);

            Xij_weighted = bsxfun(@plus, Xij_weighted - Xij_row_means - Xij_col_means, Xij_matrix_mean);

        %%% linearly combine with R_combined
            gradf = - Xij_weighted * R_combined(:);  % sign because we are minimizing negative dcov
    end

    function X_row = get_row_means(block_struct)
        % for blockproc, compute means along rows of distance matrix
        
        X_row = mean(block_struct.data,2);
        X_row = repmat(X_row, 1, size(block_struct.data,2));
    end

    function status = backtrack_check(u, f_next, gradf, t)
        % check lecture 8 of ryan tibshirani opti class
        Gt = get_Gt(u, gradf, t);

        D_uXij_t = get_D_uXij(u - t * Gt);
        status = get_f(D_uXij_t) > f_next - t * gradf' * Gt + t/2 * Gt' * Gt;
    end

    function Gt = get_Gt(u, gradf, t)
        % vector used for backtracking check with projected gradient descent
        
        u_n = u - t * gradf;
        norm_u_n = norm(u_n);
        if (norm_u_n > 1)  % project to L2 unit ball
            u_norm = u_n / norm_u_n;
        else
            u_norm = u_n;
        end

        Gt = 1/t * (u - u_norm);
    end
end








%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DCA ONE STOCH - STOCHASTIC GRADIENT DESCENT %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                   
                        
function [u, momented_gradf] = dca_one_stoch(X, Xij, R_combined, u_0, learning_rate, old_momented_grad_f, column_indices)
    % performs distance covariance analysis for one dataset and one given re-centered distance matrix
    %       uses stochastic projected gradient descent
    % 
    %   X: (N x T), data in which we want to find the N x 1 dca dimension, where
    %       N is the number of variables, and T is the number of samples
    %   Xij: (N x T^2), vector differences between X samples
    %   R_combined: (T x T), combined re-centered distance matrix of the other sets of variables
    %   u_0: (N x 1), initial guess for the dca dimension
    %   learning_rate: (1 x 1), current learning rate for stochastic gradient descent
    %   old_momented_grad_f: (N x 1), previous gradient direction (used for momentum)
    %   column_indices: (1 X T^2), used to subtract out the column means of the gradient
    %
    %  returns:
    %   u: (N x 1), the dimension of greatest distance covariance between D_X and R_combined
    %   momented_gradf

    %%% PRE-PROCESSING
        N = size(X,1);  % number of neurons
        T = size(X,2);  % number of timepoints
        
        if (sum(var(X')) < 1e-10) % X has little variability left
            u = randn(N,1);
            u = u / norm(u);
            momented_gradf = [];
            return;
        end

        u = u_0;    % set u to be initial guess
        
    %%% OPTIMIZATION
        
        % COMPUTE GRAD DESCENT PARAMETERS
        momentum_weight = 1 - learning_rate;  % momentum term convex combination of learning_rate
        D_uXij = get_D_uXij(u);  % get distance matrix of current u


        % PERFORM PROJECTED GRAD DESCENT
        gradf = get_gradf(u, D_uXij);   % worked better than Nesterov accelerated gradient
        momented_gradf = learning_rate * gradf + momentum_weight * old_momented_grad_f;
        u_unnorm = u - momented_gradf; % gradient descent step

        norm_u = norm(u_unnorm); % project u_unnorm to the L2 unit ball
        if (norm_u > 1)
            u = u_unnorm / norm_u;
        else
            u = u_unnorm;   % allow solution to exist in L2 ball (for dca_one_stoch)
        end


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % NESTED DCA_ONE_STOCH HELPER FUNCTIONS %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function D = get_D_uXij(u) 
        % compute distance matrix of (X projected onto u)
        
        D = squareform(pdist((u' * X)'));
    end

    function gradf = get_gradf(u, D_uXij)
        % computes the gradient for dca_one_stoch...note there are some tricky matrix operations in here!
        
        %%% weight Xij
            D_uXij(D_uXij == 0) = 1e-8; % add 1e-8 to avoid division by zero...other values
                                        % do not seem to matter
            % project Xij onto u
            XijT_u = Xij' * u;
        
            % weight Xij by XijT_u ./ D_uXij(:)'
            Xij_weighted = bsxfun(@times, Xij, (XijT_u ./ D_uXij(:))');

        %%% subtract row, column, and matrix means
            Xij_row_means = blockproc(Xij_weighted, [N, T], @get_row_means);
            Xij_col_means = Xij_row_means(:,column_indices);
            Xij_matrix_mean = mean(Xij_weighted,2);

            Xij_weighted = bsxfun(@plus, Xij_weighted - Xij_row_means - Xij_col_means, Xij_matrix_mean);

        %%% linearly combine with R_combined
            gradf = - Xij_weighted * R_combined(:);  % sign because we are minimizing negative dcov
    end

    function X_row = get_row_means(block_struct)
        % for blockproc, compute means along rows of distance matrix
        
        X_row = mean(block_struct.data,2);
        X_row = repmat(X_row, 1, size(block_struct.data,2));
    end
end
