function [ mdl, errors_train, errors_test ] = multilayer_backprop_HReLu_draft( X_train,Y_train, mdl, iterations,batchsize, X_test,Y_test, eta_c, eta_t, sgd_errors )
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
    A = [ ones(1, size(Xminibatch,1) ), Xminibatch ]; % ( M x D+1) = (M x D^(0)+1)
    %% Forward pass starting from the input
    for l = 1:nb_layers-1
        A = max(0, A * neural_net(l).W); 
        prop_computation(l).A = [ ones(batchsize, 1), A];
    end
    %% Back propagation
    delta_L = (2 / batchsize)*(Yminibatch - Xminibatch) .* (prop_computation(L).A > 0); % ( M x D^(L) ) = ( M x K )
    prop_computation(nb_layers).delta = delta_L; % ( M x D^(L) )
    step_down_1=-1;
    for l = nb_layers:step_down_1:2
        % get gradient matrix dV_dW^(l) for parameters W^(l) at layer l
        dV_dW_l = prop_computation(l-1).A' * prop_computation(l).delta; % (M x D ^(l-1))' x (M x D^(l)) = (D ^(l-1) x M)' x (M x D^(l))
        dV_dW_l = dV_dW_l + lambda * neural_net(l).W;
        prop_computation(l).dW = dV_dW_l; % (D ^(l-1) x D^(l))

        % compute delta for next iteration of backprop (i.e. previous layer) and threshold at 0 if O is <0 (ReLU gradient update)
        prop_computation(l-1).delta = (prop_computation(l-1).A > 0) .* prop_computation(l).delta * neural_net(l).W'; % (M x D^(l-1))
    end
    %% step size
    mod_when = 2000;
    if mod(i, mod_when) == 0
        step_size = step_size/1.2;
    end
    %% gradient step for all layers
    for j = 1:nb_layers
        neural_net(j).W = neural_net(j).W - step_size * prop_computation(j).dW;
        neural_net(j).b = neural_net(j).b - step_size * prop_computation(j).db;
    end
end

end