function [U, dcovs] = dca(X, varargin)
% function description goes here
%
% INPUTS:
%
%   X: (1 x M cell), data, M datasets
%   D (optional): (1 x N cell), distance matrices (wanting to be compared to X, but
%       not optimized, e.g., a stimulus matrix
%   
%   Additional arguments:
%       e.g. dca(X, D, 'num_iters_per_dimension', 10)
%   ('name', value)
%
%   ('num_iters_per_dimension', num_iters)
%   ('num_iters_across_datasets', num_iters)
%   ('percent_increase_criterion', percentage)
%   ('num_dca_dimensions', num_dims)
%   ('num_stoch_batch_samples', num_samples)
%   ('u_0s', u_0s)
%
% OUTPUTS:


    % PRE-PROCESSING 
    p = parse_input(X, varargin);
            % allows user to input name-value pairs
    
    check_input(p);
    
    preprocessing(p);
    
    % for each dca dim, optimize across datasets
    %      idea: optimize only one dca dim at a time (fixing all others)
    %       keep iterating until convergence
    
    % instantiate parameters
    U = cell(1,num_datasets); % cell vector, dca dimensions for each dataset
    dcovs = zeros(1,num_dca_dims); % vectors, dcovs for each dimension
    for iset = 1:num_datasets
        U_orth{iset} = eye(size(X{iset},1));  % keeps track of the orthogonal space of u
    end

    
    % OPTIMIZATION
    for idim = 1:num_dca_dims
        
        fprintf('dca dimension %d\n', idim);
        
        % initialization and pre-processing: compute the re-centered distance matrices
        %   for each dim, and update as dimensions change
        initialization(p);


        % keep track of how much we increase the total dcov
        if (p.Results.num_stoch_batch_samples > 0)
            R = cell(1,num_datasets);
            for iset = 1:num_datasets
                R{iset} = get_recentered_matrix(u{iset}, X{iset}(:,1:1000));
            end
        end
            
        total_dcov = get_total_dcov(R,D);
        total_dcov_old = total_dcov * 0.5;  % set old value to half, so it'll pass threshold
        
        itotal_dcov = 1; % keeps track of number of iterations after a run across all datasets
        best_total_dcov = 0; % keeps track of greatest dcov (for stoch grad descent)
        best_u = [];
        
        while (check_if_dcov_increases(p, total_dcov, total_dcov_old, itotal_dcov)) 
                        % if dcov does not increase by a certain percentage
                        % or if we reached the number of iterations, stop

            fprintf('  step %d: dcov = %f\n', itotal_dcov, total_dcov);
            fprintf('    sets:');
            
            r = randperm(num_datasets);  % randomize the order of datasets being optimized

            for iset = 1:num_datasets
           
                fprintf(' %d ', r(iset));
                
                if (p.Results.num_stoch_batch_samples == 0)
                    % optimize over one dataset computing full gradient
                    % with all samples
                    
                    % get combined recentered distance matrix (summed)
                    R_combined = get_recentered_combined(R((1:end)~=r(iset)), D);
                            % R_combined is T x T
                            
                    u{r(iset)} = dca_one(X{r(iset)}, Xij{r(iset)}, R_combined, u{r(iset)}, col_indices, p);
                else
                    
                    % use stochastic gradient descent with momentum
                    random_sample_indices = randperm(num_samples);
                    batch_indices = 1:p.Results.num_stoch_batch_samples:num_samples;
                    old_momented_gradf = 0;

                    for ibatch = 1:length(batch_indices)-1 % ignore last set of samples since randomized

                        window = batch_indices(ibatch):batch_indices(ibatch+1)-1;
                        sample_indices = random_sample_indices(window);

                        R = cell(1,num_datasets);
                        for iset = 1:num_datasets
                            R{iset} = get_recentered_matrix(u{iset}, X{iset}(:,sample_indices));
                        end
                        
                        R_combined = get_recentered_combined(R((1:end)~=r(iset)), D);

                        Xij = [];
                        for iset = 1:num_datasets
                            % compute all combinations of differences between samples of X
                            Xij{iset} = bsxfun(@minus, X{iset}(:,sample_indices), permute(X{iset}(:,sample_indices), [1 3 2]));
                            Xij{iset} = reshape(-Xij{iset}, size(X{iset},1), []);
                        end
                        
                        [u{r(iset)}, momented_gradf{r(iset)}] = dca_one_stoch(X{r(iset)}(:,sample_indices), ...
                            Xij{r(iset)}, R_combined, u{r(iset)}, ...
                            stoch_learning_rate, momented_gradf{r(iset)}, col_indices);

                    end

                end

                if (p.Results.num_stoch_batch_samples == 0)
                    R{r(iset)} = get_recentered_matrix(u{r(iset)}, X{r(iset)});
                end
            end
            fprintf('\n');
            
            % update dcov values
            total_dcov_old = total_dcov;
            
            if (p.Results.num_stoch_batch_samples > 0)
                R = cell(1,num_datasets);
                for iset = 1:num_datasets
                    R{iset} = get_recentered_matrix(u{iset}, X{iset}(:,1:1000));
                end
                
                % update learning rate (for stochastic gradient descent)
                stoch_learning_rate = 0.9 * stoch_learning_rate; %1 / sqrt(itotal_dcov);
            end
                        
            total_dcov = get_total_dcov(R, D);
            itotal_dcov = itotal_dcov + 1;



            if (total_dcov > best_total_dcov)
                best_total_dcov = total_dcov;
                best_u = u;
            end
            
        end
        


        % choose the best parameters seen
            total_dcov = best_total_dcov;
            u = best_u;

        
        dcovs(idim) = total_dcov;
       
        % ensure that the u are normalized
        for iset = 1:num_datasets
            u{iset} = u{iset} ./ norm(u{iset});
        end
        
        % project identified dca dimension into original space
        for iset = 1:num_datasets
            U{iset}(:,idim) = U_orth{iset} * u{iset};
        end
        
        % project data onto null space of newly found dca dimensions 
        if (idim == num_dca_dims)
            break;  % last dimension, so no need to compute null space
        end
        for iset = 1:num_datasets
            [Q,R] = qr([U{iset}(:,1:idim) ...
                    randn(size(U{iset},1), size(U{iset},1)-idim)]);
            U_orth{iset} = Q(:,(idim+1):end);

            X{iset} = U_orth{iset}' * X_orig{iset};
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
        % parses input, in case user gives name-value pairs
        
        p = inputParser;  % creates parser object
        
        default_D = [];  % distance matrices
        default_num_iters_per_dimension = 1;
        default_num_iters_across_datasets = 100;
        default_percent_increase_criterion = 0.05;  % stops when objective function increases fewer than 5% of current value
        default_num_dca_dimensions = []; % number of dca dimensions to identify
        default_u_0s = [];  % how to intialize the dimensions before optimization
        default_num_stoch_batch_samples = 0;
        
        
        addRequired(p, 'X');
        addOptional(p, 'D', default_D);
        addParamValue(p, 'num_iters_per_dimension', default_num_iters_per_dimension);
        addParamValue(p, 'num_iters_across_datasets', default_num_iters_across_datasets);
        addParamValue(p, 'percent_increase_criterion', default_percent_increase_criterion);
        addParamValue(p, 'num_dca_dimensions', default_num_dca_dimensions);
        addParamValue(p, 'num_stoch_batch_samples', default_num_stoch_batch_samples);
        addParamValue(p, 'u_0s', default_u_0s);
        
        % NOTE: addParamValue should be changed to addParameter...I have to
        % use addParamValue because the ECE cmu cluster hasn't updated
        % their matlab software...
        
        parse(p,X,vargs{:});  % parses input to get optional parameters
            % now p.Results.X, etc.
        
    end


    function check_input(p)
        % make sure user input correct formats
        
        % check X
            if (~iscell(p.Results.X) || size(p.Results.X,1) > 1 && size(p.Results.X,2) > 1) % check if X is a cell vector  
                error('X (1 x num_datasets) should be a cell array, where X{iset} is (num_variables x num_datapoints)');
            end
            
            num_datasets = length(p.Results.X);
            
            [num_vars, num_datapoints] = cellfun(@size, p.Results.X);
            if (length(unique(num_datapoints)) ~= 1)  % there should only be one sample size
                error('Dataset(s) in X do not contain the same number of datapoints. X{iset} (num_variables x num_datapoints), where num_datapoints is the same for each dataset (but num_variables can be different).');
            end
            
        % check D
            if (~isempty(p.Results.D) && (~iscell(p.Results.D) || size(p.Results.D, 1) > 1 && size(p.Results.D, 2) > 1))
                error('Dataset(s) in D do not contain the same number of datapoints. D{iset} (num_datapoints x num_datapoints) are distance matrices, where num_datapoints is the same for each dataset.');
            end
            
        % check that X and D have more than just one dataset
            if (length(p.Results.X) + length(p.Results.D) <= 1)
                error('Not enough datasets in X and D.  The number of datasets (including distance matrices) should be at least two.');
            end
            
        % check num_dca_dimensions
            if (p.Results.num_dca_dimensions > min(num_vars))
                error(sprintf('"num_dca_dimensions" must be less than or equal to %d, the minimum number of variables across datasets.', min(num_vars)));
            end
       
    end


    function preprocessing(p)
    %   - compute any fixed variables before optimization
    %   - initialize any needed numbers
    
        X = p.Results.X;
        X_orig = X;
        
        % find the number of datasets
        num_datasets = length(X);
        
        % check how many dca dimensions there should be
        %   for minimum number of dimensions across datasets
        %   + user input
            num_dims_foreach_set = [];
            for iset = 1:num_datasets
                num_dims_foreach_set = [num_dims_foreach_set size(X{iset},1)];
            end
            num_dca_dims = min(num_dims_foreach_set);

            if (~isempty(p.Results.num_dca_dimensions))
                num_dca_dims = p.Results.num_dca_dimensions;
            end
        
        % find the number of samples
        num_samples = size(X{1},2);
        

        % compute the recentered matrices for user input D
        D = p.Results.D;
        if (~isempty(D))
            for iset = 1:length(D)
                H = eye(size(D{iset})) - 1 / size(D{iset},1) * ones(size(D{iset}));
                D{iset} = H * D{iset} * H;  % recenters matrix
            end
        end
        

        % prepare indices for column indices when subtracting off column means
        
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

    end


    function initialization(p)
    %   initialize U, U_orth, and dcovs
    %   U_orth keeps track of the null space of U
        

        for iset = 1:num_datasets
            % initialize the weights of the dca dimensions either
            %   randomly or with user's choice
            if (~isempty(p.Results.u_0s) && size(p.Results.u_0s{iset},1) == size(X{iset},1))  % if user input initialized 
                u{iset} = p.Results.u_0s{iset}(:,1);
            else
                u{iset} = orth(randn(size(X{iset},1), 1));
            end
        end

        % get initial recentered matrices for each dataset
            if (p.Results.num_stoch_batch_samples == 0)  % only for full grad descent
                R = cell(1,num_datasets);
                for iset = 1:num_datasets
                    R{iset} = get_recentered_matrix(u{iset}, X{iset});
                end
            end
        
        % stochastic gradient descent initialization
            if (p.Results.num_stoch_batch_samples > 0)
                stoch_learning_rate = 1;  % initial learning rate for SGD
                momented_gradf = cell(1,num_datasets);
                for iset = 1:num_datasets
                    momented_gradf{iset} = zeros(size(u{iset}));
                end
            end
       
        
        % compute Xij for each dataset
        if (p.Results.num_stoch_batch_samples == 0)  % only for full grad descent
            Xij = [];
            for iset = 1:num_datasets
                % compute all combinations of differences between samples of X
                Xij{iset} = bsxfun(@minus, X{iset}, permute(X{iset}, [1 3 2]));
                Xij{iset} = reshape(-Xij{iset}, size(X{iset},1), []);
            end
        end
    end



end





%%%%%%%%%%%%%%%%%%%%%%%
%  NON-NESTED FUCNTIONS
%%%%%%%%%%%%%%%%%%%%%%%


function R = get_recentered_matrix(u, X)
    % computes the recentered distance matrix for each dataset
    % u: num_variables x 1
    % X: num_variables x num_datapoints

    % compute distance matrix of (X projected onto u
        D = squareform(pdist((u' * X)'));
        
    % now recenter it
        H = eye(size(D)) - 1/size(D,1) * ones(size(D));
        R = H * D * H;  % recenters distance matrix
end

        % compute distance matrix of (X projected onto u)

        
        
        

function total_dcov = get_total_dcov(R,D)
% compute the total distance covariance across all datasets

    R = [R D];
    
    Rtotal = 0;
    T = size(R{1},1);
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
                    itotal_dcov <= p.Results.num_iters_across_datasets)
            result = true;
        else
            result = false;
        end
    else  % stochastic gradient descent...just check number of iterations
        if (itotal_dcov <= p.Results.num_iters_across_datasets)
            result = true;
        else
            result = false;
        end
    end

end



function R_combined = get_recentered_combined(R, D)
    % compute the combined matrix of all recentered distance matrices
    % returns a matrix, where each element is a pointwise-sum of all R
    % and D

    if (~isempty(R))
        R_combined = zeros(size(R{1}));
    else
        R_combined = zeros(size(D{1}));
    end
    
    for iset = 1:length(R)
        R_combined = R_combined + R{iset};
    end
    
    for iset = 1:length(D)
        R_combined = R_combined + D{iset};
    end
    
    R_combined = R_combined / (length(R) + length(D));

end




function u = dca_one(X, Xij, R_combined, u_0, column_indices, p)
%
% [u, R] = dca_one(X, R_combined, u_0, p)
%
% performs distance covariance analysis for one dataset and one
% 
%
% INPUT:
%   X: (N x T), data in which we want to find the N x 1 dca dimension, where
%       N is the number of variables, and T is the number of observations
%   R_combined: (T x T), distance matrix of the other sets of variables
%   u_0: (N x 1), initial guess for the dca dimension
%   p: (1 x 1), inputParser object which contains user constraints, such
%               as number of iterations
%
% OUTPUT:
%   u: (N x 1), the dimension of 
%           greatest distance covariance between D_X and R_combined
%   R: (N x N), distance matrix of X with u
%



    N = size(X,1);  % number of neurons
    T = size(X,2);  % number of timepoints

    % PRE-PROCESSING

    if (sum(var(X')) < 1e-10) % X has little variability left
        u = randn(N,1);
        u = u / norm(u);
        return;
    end


    % OPTIMIZATION
    u = u_0;    % set u to be initial guess

    for istep = 1:p.Results.num_iters_per_dimension  % stop when num iters have been reached

        % COMPUTE GRAD DESCENT PARAMETERS
        D_uXij = get_D_uXij(u);  % get distance matrix of current u
        f_val = get_f(D_uXij);     % compute current function value and gradf for backtracking
        gradf = get_gradf(u, D_uXij);

        t = 1;  % backtracking step size

        % first check large intervals for t (so backtracking loop doesn't
        % take forever)
        for candidate_power = 1:9
            if (~backtrack_check(u, f_val, gradf, 10^-candidate_power))
                break;
            else
                t = 10^-candidate_power;
            end
        end

        
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
            u = u_unnorm;   % need to consider cases inside the L2 unit ball as well (to make convex candidate set)
        end

    end




    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % NESTED DCA_ONE HELPER FUNCTIONS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function D = get_D_uXij(u) 
        % compute distance matrix of (X projected onto u)
        
        D = squareform(pdist((u' * X)'));
       
    end


    function f = get_f(D_uXij)
        H = eye(T) - 1/T * ones(T);
        A = H * D_uXij * H;  % recenters distance matrix
        f = -sum(sum(R_combined .* A));
    end


    function gradf = get_gradf(u, D_uXij)

        % first weight each i,jth element of XijXijT by 1/distance
        D_uXij(D_uXij == 0) = 1e-8;  % add 1e-8 to avoid division by zero
        
        % project Xij onto u
        XijT_u = Xij' * u;
        
        % weight Xij by XijT_u ./ D_uXij(:)'
        Xij_weighted = bsxfun(@times, Xij, (XijT_u ./ D_uXij(:))');
        

        % subtract row, column, and matrix means
        Xij_row_means = blockproc(Xij_weighted, [N, T], @get_row_means);
        Xij_col_means = Xij_row_means(:,column_indices);
        Xij_matrix_mean = mean(Xij_weighted,2);
        
        Xij_weighted = bsxfun(@plus, Xij_weighted - Xij_row_means - Xij_col_means, Xij_matrix_mean);
        

        % now need to linearly combine with B (similar way as with D)
        gradf = - Xij_weighted * R_combined(:);

    end



    function X_row = get_row_means(block_struct)
        X_row = mean(block_struct.data,2);
        X_row = repmat(X_row, 1, size(block_struct.data,2));
    end



    function Gt = get_Gt(u, gradf, t)
        u_n = u - t * gradf;
        norm_u_n = norm(u_n);
        if (norm_u_n > 1)  % project to L2 unit ball
            u_norm = u_n / norm_u_n;
        else
            u_norm = u_n;
        end

        Gt = 1/t * (u - u_norm);
    end


    function status = backtrack_check(u, f_next, gradf, t)
        % lecture 8 of ryan tibs opti class
        Gt = get_Gt(u, gradf, t);

        D_uXij_t = get_D_uXij(u - t * Gt);
        status = get_f(D_uXij_t) > f_next - t * gradf' * Gt + t/2 * Gt' * Gt;
    end


end







%%% DCA_STOCH...for stochastic gradient descent                        
                        
function [u, momented_gradf] = dca_one_stoch(X, Xij, R_combined, u_0, learning_rate, old_momented_grad_f, column_indices)
%
% [u, R] = dca_one(X, R_combined, u_0, p)
%
% performs distance covariance analysis for one dataset and one other
% re-centered distance matrix
% 
%
% INPUT:
%   X: (N x T), data in which we want to find the N x 1 dca dimension, where
%       N is the number of variables, and T is the number of observations
%   R_combined: (T x T), distance matrix of the other sets of variables
%   u_0: (N x 1), initial guess for the dca dimension
%   p: (1 x 1), inputParser object which contains user constraints, such
%               as number of iterations
%
% OUTPUT:
%   u: (N x 1), the dimension of 
%           greatest distance covariance between D_X and R_combined
%   R: (N x N), distance matrix of X with u
%



    N = size(X,1);  % number of neurons
    T = size(X,2);  % number of timepoints

    % PRE-PROCESSING
    % compute all combinations of differences and their outer-product 

    if (sum(var(X')) < 1e-10) % X has little variability left
        u = randn(N,1);
        u = u / norm(u);
        R = get_D_uXij(u);

        return;
    end


    % OPTIMIZATION
    u = u_0;    % set u to be initial guess


    % COMPUTE GRAD DESCENT PARAMETERS
    D_uXij = get_D_uXij(u);  % get distance matrix of current u
    gradf = get_gradf(u, D_uXij);

    
    % PERFORM PROJECTED GRAD DESCENT
    momentum_weight = 1 - learning_rate;  % momentum term convex combination of learning_rate
    momented_gradf = learning_rate * gradf + momentum_weight * old_momented_grad_f;
    u_unnorm = u - momented_gradf; % gradient descent step
    

    norm_u = norm(u_unnorm); % project u_unnorm to the L2 unit ball
    if (norm_u > 1)
        u = u_unnorm / norm_u;
    else
        u = u_unnorm;   % need to consider cases inside the L2 unit ball as well (to make convex candidate set)
    end



    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % NESTED DCA_ONE_STOCH HELPER FUNCTIONS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function D = get_D_uXij(u) 
        % compute distance matrix of (X projected onto u)
        
        D = squareform(pdist((u' * X)'));
        
    end



    function gradf = get_gradf(u, D_uXij)

        % first weight each i,jth element of XijXijT by 1/distance
        D_uXij(D_uXij == 0) = 1e-8;  % add 1e-8 to avoid division by zero
        
        % project Xij onto u
        XijT_u = Xij' * u;
        
        % weight Xij by XijT_u ./ D_uXij(:)'
        Xij_weighted = bsxfun(@times, Xij, (XijT_u ./ D_uXij(:))');
        

        % subtract row, column, and matrix means
        Xij_row_means = blockproc(Xij_weighted, [N, T], @get_row_means);

        Xij_col_means = Xij_row_means(:,column_indices);
        Xij_matrix_mean = mean(Xij_weighted,2);
        
        
        Xij_weighted = bsxfun(@plus, Xij_weighted - Xij_row_means - Xij_col_means, Xij_matrix_mean);
        

        % now need to linearly combine with B (similar way as with D)
        gradf = - Xij_weighted * R_combined(:);

    end



    function X_row = get_row_means(block_struct)
        X_row = mean(block_struct.data,2);
        X_row = repmat(X_row, 1, size(block_struct.data,2));
    end


end
