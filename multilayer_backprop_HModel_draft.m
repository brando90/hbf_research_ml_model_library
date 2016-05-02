function [ nn_mdl, errors_train, errors_test ] = multilayer_backprop_HModel_draft( X_train,Y_train, nn_mdl, iterations,batchsize, X_test,Y_test, eta_c, eta_t, sgd_errors )
fprintf('sgd_errors = %d',sgd_errors);
[N, ~] = size(X_train);
[~,D_out] = size(Y_train);
L = nb_layers;
if sgd_errors
    errors_train = zeros(iterations+1,1);
    errors_test = zeros(iterations+1,1);
    errors_train(1) = compute_Hf_sq_error(X_train,Y_train, nn_mdl, nn_mdl.lambda);
    errors_test(1) = compute_Hf_sq_error(X_test,Y_test, nn_mdl, nn_mdl.lambda);
end
% G_c = ones(K, D_out);
% G_t = ones(D, K);
for i=2:length(errors_test)
    %% get minibatch
    mini_batch_indices = ceil(rand(batchsize,1) * N); % M
    Xminibatch =  X_train(mini_batch_indices,:); % ( M x D ) =( M x D^(0) )
    Yminibatch = Y_train(mini_batch_indices,:); % ( M x D^(L) )
    A = Xminibatch; % ( M x D+1) = (M x D^(0)+1)
    %% Forward pass starting from the input
    for l = 1:L-1
        A = nn_mdl.Act( [ones(batchsize,1), A] * neural_net(l).W ); % (M x D^(l)) = (M x D^(l-1)+1) x (D^(l-1)+1 x D^(l))
        fp(l).A = A; % (M x D^(l))
    end
    %% Back propagation
    delta_L = (2 / batchsize)*(fp(L).A - Yminibatch) .* nn_mdl(L).dAct_ds( fp(L).A ); % ( M x D^(L) ) = (M x D^(L)) .* (M x D^(L))
    backprop(L).delta = delta_L; % ( M x D^(L) )
    step_down_1=-1;
    for l = L:step_down_1:2
        % get gradient matrix dV_dW^(l) for parameters W^(l) at layer l
        dV_dW_l = [ones(batchsize,1), fp(l-1).A]' * backprop(l).delta; % (D^(l-1)+1 x D^(l)) = (M x D ^(l-1)+1)' x (M x D^(l))
        D_l_1 = size(fp(l-1).A,1); % get value of D^(l-1)
        dV_dW_l(2:D_l_1+1,:) = dV_dW_l + lambda * nn_mdl(l).W(2:D_l_1+1,:); % regularize everything except the offset
        backprop(l).dW = dV_dW_l; % (D ^(l-1)+1 x D^(l))

        % compute delta for next iteration of backprop (i.e. previous layer) and threshold at 0 if O is <0 (ReLU gradient update)
        backprop(l-1).delta = nn_mdl.dAct_ds( fp(l-1).A ) .* backprop(l).delta * nn_mdl(l).W'; % (M x D^(l-1)) = (M x D^(l) x ()
    end
    %% step size
    mod_when = 2000;
    if mod(i, mod_when) == 0
        step_size = step_size/1.2;
    end
    %% gradient step for all layers
    for j = 1:nb_layers
        nn_mdl(j).W = nn_mdl(j).W - step_size * backprop(j).dW;
    end
end
end