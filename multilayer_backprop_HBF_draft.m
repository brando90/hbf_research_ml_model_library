function [ hbf_net, errors_train, errors_test ] = multilayer_backprop_HBF_draft( X_train,Y_train, hbf_net, iterations,batchsize, X_test,Y_test, eta_c, eta_t, sgd_errors )
fprintf('sgd_errors = %d',sgd_errors);
[N, ~] = size(X_train);
[~,D_out] = size(Y_train);
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
    A = Xminibatch; % ( M x D+1) = (M x D^(0)+1)
    %% Forward pass starting from the input
    for l = 1:nb_layers-1
        WW = sum(hbf_net(l).W .* hbf_net(l).W, 1); % ( 1 x D^(l) )
        XX = sum(A .* A, 2); % (M x 1)
        moving_offset = bsxfun(@plus, XX, WW); % ( M x D^(l) )
        Z = 2 * hbf_net.beta * A*hbf_net(l).W - (hbf_net.beta * moving_offset); % (M x D^(l)) = (M x D^(l-1)+1) x (D^(l-1)+1 x D^(l))
        prop_computation(l).A = exp(Z); % (M x D^(l))
    end
    %% Back propagation
    A_L = prop_computation(nb_layers).A;
    delta_L = (2 / batchsize)*( A_L - Yminibatch ) .* hbf_net(nb_layers).dAct_ds( A_L ); % ( M x D^(L) ) = (M x D^(L)) .* (M x D^(L))
    prop_computation(nb_layers).delta = delta_L; % ( M x D^(L) )
    step_down_1=-1;
    for l = nb_layers:step_down_1:2
        % get gradient matrix dV_dW^(l) for parameters W^(l) at layer l
        dV_dW_l = [ones(batchsize,1), prop_computation(l-1).A]' * prop_computation(l).delta; % (D^(l-1)+1 x D^(l)) = (M x D ^(l-1)+1)' x (M x D^(l))
        D_l_1 = size(prop_computation(l-1).A,1); % get value of D^(l-1)
        dV_dW_l(2:D_l_1+1,:) = dV_dW_l + lambda * hbf_net(l).W(2:D_l_1+1,:); % regularize everything except the offset
        prop_computation(l).dW = dV_dW_l; % (D ^(l-1)+1 x D^(l))

        % compute delta for next iteration of backprop (i.e. previous layer) and threshold at 0 if O is <0 (ReLU gradient update)
        prop_computation(l-1).delta = (prop_computation(l-1).A > 0) .* prop_computation(l).delta * hbf_net(l).W'; % (M x D^(l-1)) = (M x D^(l) x ()
    end
    %% step size
    mod_when = 2000;
    if mod(i, mod_when) == 0
        step_size = step_size/1.2;
    end
    %% gradient step for all layers
    for j = 1:nb_layers
        hbf_net(j).W = hbf_net(j).W - step_size * prop_computation(j).dW;
        hbf_net(j).b = hbf_net(j).b - step_size * prop_computation(j).db;
    end
end
end