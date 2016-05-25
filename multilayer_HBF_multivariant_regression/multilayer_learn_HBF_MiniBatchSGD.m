function [mdl, errors_train, errors_test] = multilayer_learn_HBF_MiniBatchSGD( X_train,Y_train, mdl, nb_iterations,batchsize, X_test,Y_test, step_size_params, sgd_errors )
fprintf('sgd_errors = %d \n', sgd_errors);
[N, ~] = size(X_train);
[~,D_out] = size(Y_train);
L = size(mdl,2);
if sgd_errors
    errors_train = zeros(nb_iterations+1,1);
    errors_test = zeros(nb_iterations+1,1);
    errors_train(1) = compute_Hf_sq_error(X_train,Y_train, mdl);
    errors_test(1) = compute_Hf_sq_error(X_test,Y_test, mdl);
end
if step_size_params.AdaGrad
    G_c = ones(K, D_out);
    G_t = ones(D, K);
elseif step_size_params.Decaying
    step_size = step_size_params.step_size; %TODO
end
fp = struct('A', cell(1,L), 'Z', cell(1,L), 'Delta_tilde', cell(1,L));
backprop = struct('delta', cell(1,L), 'dW', cell(1,L), 'db', cell(1,L));

fprintf ('Iter %d. Training zero-one error: %f; Testing zero-one error: %f; step size =%f \n', 0, errors_train(1), errors_test(1), step_size)

for i=2:length(errors_test)
    %% get minibatch
    mini_batch_indices = ceil(rand(batchsize,1) * N); % M
    Xminibatch =  X_train(mini_batch_indices,:); % ( M x D ) =( M x D^(0) )
    Yminibatch = Y_train(mini_batch_indices,:); % ( M x D^(L) )
    %% Forward pass starting from the input
    L = size(mdl,2);
    A = Xminibatch; % ( M x D) = (M x D^(0))
    for l = 1:L-1
        WW = sum(mdl(l).W.^2, 1); % ( 1 x D^(l-1)= sum( (M x D^(l)), 1 )
        XX = sum(A.^2, 2); % (M x 1) = sum( (M x D^(l-1)), 2 )
        Delta_tilde = 2*(A*mdl(l).W) - bsxfun(@plus, WW, XX) ;
        %Z = mdl(l).beta*( 2*(A*mdl(l).W) - bsxfun(@plus, WW, XX)); % (M x D^(l-1)) - (M x D^(l-1))
        Z = mdl(l).beta*( Delta_tilde ); % (M x D^(l-1)) - (M x D^(l-1))
        A = mdl(l).Act(Z); % (M x D^(l))
        fp(l).Delta_tilde = Delta_tilde;
        fp(l).Z = Z;
        fp(l).A = A; % (M x D^(l))
    end
    % activation for final layer
    if mdl(L).Act( ones(1) ) == ones(1) %% if Act func is Identity i.e. NO ACTIVATION for final Layer
        A = mdl(L).Act( A * mdl(L).W ); % (M x D^(l)) = (M x D^(l-1)+1) x (D^(l-1)+1 x D^(l))
        fp(L).A = A; % (M x D^(l))
    else
        WW = sum(mdl(L).W.^2, 1); % ( 1 x D^(l-1)= sum( (M x D^(l)), 1 )
        XX = sum(A.^2, 2); % (M x 1) = sum( (M x D^(l-1)), 2 )
        Delta_tilde = 2*(A*mdl(L).W) - bsxfun(@plus, WW, XX) ;
        %Z = mdl(l).beta*( 2*(A*mdl(l).W) - bsxfun(@plus, WW, XX)); % (M x D^(l-1)) - (M x D^(l-1))
        Z = mdl(L).beta*( Delta_tilde ); % (M x D^(l-1)) - (M x D^(l-1))
        A = mdl(L).Act(Z); % (M x D^(l))
        fp(L).Delta_tilde = Delta_tilde;
        fp(L).Z = Z;
        fp(L).A = A; % (M x D^(l))
    end
        
    %% Back propagation dJ_dW
    backprop(L).delta = (2 / batchsize)*( fp(L).A - Yminibatch ) .* mdl(L).dAct_ds( fp(L).A ); % ( M x D^(L) ) = (M x D^(L)) .* (M x D^(L))
    for l = L:-1:2
        if l == L && mdl(L).Act( ones(1) ) == ones(1)
            % get gradient matrix dV_dW^(l) for parameters W^(l) at layer l
            backprop(l).dW = fp(l-1).A' * backprop(l).delta; % (D^(l-1) x D^(l)) = (M x D ^(l-1))' x (M x D^(l))
            % compute delta for next iteration of backprop (i.e. previous layer)
            backprop(l-1).delta = mdl(l-1).dAct_ds(fp(l-1).A) .* (backprop(l).delta * mdl(l).W');
        else
            % get gradient matrix dV_dW^(l) for parameters W^(l) at layer l
            T_ijm = bsxfun( @times, mdl(l).W, reshape(backprop(l).delta',[1,flip( size(backprop(l).delta) )] ) ); % ( D^(l - 1) x D^(l) x M )
            backprop(l).dW = 2 * mdl(l).beta * ( fp(l-1).A'*backprop(l).delta - sum( T_ijm, 3) ); % (D^(l-1) x D^(l)) = (D^(l-1) x D^(l)) .- sum[ (D^(l-1) x D^(l) x M), 3 ] = (D^(l-1) x M) x (M x D^(l)) .- sum[ (D^(l-1) x D^(l) x M), 3 ]
            backprop(l).dBeta = sum( sum(backprop(l).delta .* fp(l).Delta_tilde) ); % (1 x 1)  = sum(sum( (N x D^(l)) .x (N x D^(l)) ))
            % compute delta for next iteration of backprop (i.e. previous layer)
            delta_sum = sum(backprop(l).delta ,2); % (M x 1) <- sum( (M x D^(l)), 2 ) 
            A_x_delta = bsxfun(@times, fp(l-1).A, delta_sum); % (M x D^(L)) = (M x D^(l)) .* (M x 1)
            backprop(l-1).delta = 2*mdl(l).beta * mdl(l-1).dAct_ds( fp(l-1).A ).*( backprop(l).delta*mdl(l).W' - A_x_delta ); % (M x D^(l-1))
        end
    end
    l=1;
    T_ijm = bsxfun( @times, mdl(l).W, reshape(backprop(l).delta',[1,flip( size(backprop(l).delta) )] ) ); % ( D^(l - 1) x D^(l) x M )
    backprop(l).dW = 2 * mdl(l).beta * ( Xminibatch'*backprop(l).delta - sum( T_ijm, 3) ); % (D^(l-1) x D^(l)) = (D^(l-1) x D^(l)) .- sum[ (D^(l-1) x D^(l) x M), 3 ] = (D^(l-1) x M) x (M x D^(l)) .- sum[ (D^(l-1) x D^(l) x M), 3 ]
    backprop(l).dBeta = sum( sum(backprop(l).delta .* fp(l).Delta_tilde) ); %  = sum( (N x ) .x () )
    %% step size
    mod_when = step_size_params.mod_when;
    if step_size_params.AdaGrad
        G_c = G_c + dJ_dc.^2;
        mu_c = step_size_params.eta_c ./ ( (G_c).^0.5 );
        G_t = G_t + dJ_dt.^2;
        mu_t = step_size_params.eta_t ./ ( (G_t).^0.5 );
    elseif step_size_params.Decaying
        if mod(i, mod_when) == 0
            step_size = step_size/step_size_params.decay_rate;
        end
    end
    
    %% gradient step for all layers
    for j = 1:L
        mdl(j).W = mdl(j).W - step_size * backprop(j).dW;
        mdl(j).beta = mdl(j).beta - step_size * backprop(j).dBeta;
    end
    %% print errors
    if sgd_errors
        errors_train(i) = compute_Hf_sq_error_vec(X_train,Y_train, mdl);
        errors_test(i) = compute_Hf_sq_error_vec(X_test,Y_test, mdl);
        %if mod(i, ceil(nb_iterations/100)) == 0 && step_size_params.print_error_to_screen
        print_every_multiple = step_size_params.print_every_multiple;
        if mod(i, print_every_multiple) == 0 && step_size_params.print_error_to_screen
            % Display the results achieved so far
            fprintf ('Iter %d. Training zero-one error: %f; Testing zero-one error: %f; step size = %f \n', i, errors_train(i), errors_test(i), step_size)
        end
    end
end
end