function [ mdl, errors_train, errors_test ] = multilayer_backprop_HBF_draft( X_train,Y_train, mdl, iterations,batchsize, X_test,Y_test, eta_c, eta_t, sgd_errors )
fprintf('sgd_errors = %d',sgd_errors);
[N, ~] = size(X_train);
[~,D_out] = size(Y_train);
L = size(mdl,2);
if sgd_errors
    errors_train = zeros(iterations+1,1);
    errors_test = zeros(iterations+1,1);
    errors_train(1) = compute_Hf_sq_error(X_train,Y_train, mdl, mdl.lambda);
    errors_test(1) = compute_Hf_sq_error(X_test,Y_test, mdl, mdl.lambda);
end
% G_c = ones(K, D_out);
% G_t = ones(D, K);
for i=2:length(errors_test)
    %% get minibatch
    mini_batch_indices = ceil(rand(batchsize,1) * N); % M
    Xminibatch =  X_train(mini_batch_indices,:); % ( M x D ) =( M x D^(0) )
    Yminibatch = Y_train(mini_batch_indices,:); % ( M x D^(L) )
    A = Xminibatch; % ( M x D) = (M x D^(0))
    %% Forward pass starting from the input
    for l = 1:L-1
        WW = sum(mdl(l).W .* mdl(l).W, 1); % ( 1 x D^(l) )
        XX = sum(A .* A, 2); % (M x 1)
        moving_offset = bsxfun(@plus, XX, WW); % ( M x D^(l) ) = (M x 1) .+ (1 x D^(l))
        Z = 2 * mdl.beta ( A*mdl(l).W - moving_offset); % (M x D^(l)) = (M x D^(l-1)+1) x (D^(l-1)+1 x D^(l))
        A = mdl.Act(Z); % (M x D^(l))
        fp(l).A = A; % (M x D^(l))
    end
    % activation for final layer
    if mdl(L).Act( ones(1) ) == ones(1) %% if Act func is Identity 
        A = mdl(L).Act( A * neural_net(L).W ); % (M x D^(l)) = (M x D^(l-1)+1) x (D^(l-1)+1 x D^(l))
        fp(L).A = A; % (M x D^(l))
    else
        WW = sum(mdl(L).W .* mdl(L).W, 1); % ( 1 x D^(l) )
        XX = sum(A .* A, 2); % (M x 1)
        moving_offset = bsxfun(@plus, XX, WW); % ( M x D^(l) ) = (M x 1) .+ (1 x D^(l))
        Z = 2 * mdl.beta ( A*mdl(L).W - moving_offset); % (M x D^(l)) = (M x D^(l-1)+1) x (D^(l-1)+1 x D^(l))
        fp(L).A = mdl(L).Act(Z); % (M x D^(l))
    end
    %% Back propagation
    backprop(L).delta = (2 / batchsize)*( fp(L).A - Yminibatch ) .* mdl(L).dAct_ds( fp(L).A ); % ( M x D^(L) ) = (M x D^(L)) .* (M x D^(L))
    step_down_1=-1;
    for l = L:step_down_1:2
        % get gradient matrix dV_dW^(l) for parameters W^(l) at layer l
        T_ijm = bsxfun( @times, W, reshape(backprop(l).delta',[1,flip( size(backprop(l).delta) )] ) ); % ( D^(l - 1) x D^(l) x M )
        backprop(l).dW = 2 * mdl.beta * ( fp(l-1).A'*backprop(l).delta - sum( T_ijm, 3) ); % (D^(l-1) x D^(l)) = (D^(l-1) x D^(l)) .- sum[ (D^(l-1) x D^(l) x M), 3 ] = (D^(l-1) x M) x (M x D^(l)) .- sum[ (D^(l-1) x D^(l) x M), 3 ]
        
        % compute delta for next iteration of backprop (i.e. previous layer) and threshold at 0 if O is <0 (ReLU gradient update)
        delta_sum = sum(backprop(l).delta ,2); % (M x 1) <- sum( (M x L), 2 ) 
        A_delta = bsxfun(@times, fp(l).A, delta_sum); % (M x L) = (M x L) .* (M x 1)
        backprop(l-1).delta = 2*mdl.beta * hbf_netdAct_ds( fp(l-1).A ).*( backprop(l).delta*mdl(l).W' - A_delta ); % (M x D^(l-1)) = (M x D^(l) x ()
    end
    %% step size
    mod_when = 2000;
    if mod(i, mod_when) == 0
        step_size = step_size/1.2;
    end
    %% gradient step for all layers
    for j = 1:L
        mdl(j).W = mdl(j).W - step_size * backprop(j).dW;
    end
end
end