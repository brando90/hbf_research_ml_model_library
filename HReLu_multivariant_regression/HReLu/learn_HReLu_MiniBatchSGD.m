function [ mdl, errors_train, errors_test ] = learn_HReLu_MiniBatchSGD(X_train,Y_train, mdl, iterations, X_test,Y_test, step_size_params, sgd_errors)
fprintf('sgd_errors = %d',sgd_errors);
[~, K] = size(mdl.t);
[N, D] = size(X_train);
[~,D_out] = size(Y_train);
if sgd_errors
    errors_train = zeros(iterations+1,1);
    errors_test = zeros(iterations+1,1);
    errors_train(1) = compute_Hf_sq_error(X_train,Y_train, mdl, mdl.lambda);
    errors_test(1) = compute_Hf_sq_error(X_test,Y_test, mdl, mdl.lambda);
end
if step_size_params.AdaGrad
    G_c = ones(K, D_out);
    G_t = ones(D, K);
elseif step_size_params.Decaying
    step_size = step_size/1.2;
end
for i=2:length(errors_test)
    mini_batch_indices = ceil(rand(batchsize,1) * nb_examples);
    Xminibatch =  X_train(mini_batch_indices,:);
    Yminibatch = Y_train(mini_batch_indices,:);
    %% Forward pass
    A_0 = Xminibatch; % (M x D+1)
    Z = [ones(batchsize,1), A_0] * obj.t;  % (M * K) = (M x D+1) x (D+1 * K)
    A = max(0, Z); % (M * K)
    F_X = A * obj.c; % (M x D) = (M * K) * (K * D)
    %% step size
    if step_size_params.AdaGrad
        G_c = G_c + dJ_dc.^2;
        mu_c = step_size_params.eta_c ./ ( (G_c).^0.5 );
        G_t = G_t + dJ_dt.^2;
        mu_t = step_size_params.eta_t ./ ( (G_t).^0.5 );
    elseif step_size_params.Decaying
        if mod(i, mod_when) == 0
            step_size = step_size/1.2;
        end
    end
    %% gradients * backprop
    delta_L2 = (2/batchsize) * (F_X - Yminibatch) .* (F_X > 0); % ( M x D^(L2) )
    dJ_dc = [ones(batchsize,1),A_L1]' * delta_L2; % (D^(L1)+1 x D^(L2)) = (M x D ^(L1)+1)' x (M x D^(l))
    delta_L1 = (A_L1 > 0) .* delta_L2 * c'; % (M x D^(L1)) = (M x D^(L1)) .* (M x D^(L2)) x (D^(L1) x D^(L2))'
    dJ_dt = A_0' * delta_L1; % (D^(L0)+1 x D^(L1)) = (M x D^(L0))' x (M x D^(L1))
    %% update mdl params
    mdl.c = mdl.c - mu_c .* (dJ_dc + lambda * neural_net(l).W);
    mdl.t = mdl.t - mu_t .* (dJ_dt + lambda * neural_net(l).W);
    if sgd_errors
        errors_train(i) = compute_Hf_sq_error(X_train,Y_train, mdl_new, mdl.lambda);
        errors_test(i) = compute_Hf_sq_error(X_test,Y_test, mdl_new, mdl.lambda);
    end
    if rem(i,500) == 0
        fprintf('sgd iteration = %d\n',i);
        fprintf('errorTRAIN(%d) = %d\n', i, errors_train(i));
        fprintf('errorTEST(%d) = %d\n', i, errors_test(i));
    end
end
end