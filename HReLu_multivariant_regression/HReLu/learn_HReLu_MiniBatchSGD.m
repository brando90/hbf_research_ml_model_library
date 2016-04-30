function [ mdl, errors_train, errors_test ] = learn_HReLu_MiniBatchSGD(X_train,Y_train, mdl, iterations, X_test,Y_test, eta_c, eta_t, sgd_errors)
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
G_c = ones(K, D_out);
G_t = ones(D, K);
for i=2:length(errors_test)
    mini_batch_indices = ceil(rand(batchsize,1) * nb_examples);
    Xminibatch =  X_train(mini_batch_indices,:);
    Yminibatch = Y_train(mini_batch_indices,:);
    [F_X, Z_L1, A_L1] = mdl.f(Xminibatch);
    %% step size
    [mu_c, G_c_new] = choose_step_size_c_adagrad(eta_c, dJ_dc, G_c);
    [mu_t, G_t_new] = choose_step_size_t_adagrad(eta_t, dJ_dt, G_t);
    %% gradients * backprop
    delta_L2 = (2/batchsize) * (F_X - Yminibatch) .* (F_X > 0); % ( M x D^(L) )
    dJ_dc = A_L1' * delta_L2;
    delta_L1 = (A_L1 > 0) .* delta_L2 * c'; % (M x D^(L-1)) = (M x D^(L-1)) .* (M x D^(L)) x (D^(L-1) x D^(L-1))
    dJ_dt = Xminibatch' * delta_L1;
    %% update
    c_new = mdl.c - mu_c .* (dJ_dc + 0 * mdl.lambda);
    t_new = mdl.t - mu_t .* (dJ_dt + 0 * mdl.lambda);
    if sgd_errors
        errors_train(i) = compute_Hf_sq_error(X_train,Y_train, mdl_new, mdl.lambda);
        errors_test(i) = compute_Hf_sq_error(X_test,Y_test, mdl_new, mdl.lambda);
    end
    if rem(i,500) == 0
        fprintf('sgd iteration = %d\n',i);
        fprintf('errorTRAIN(%d) = %d\n', i, errors_train(i));
        fprintf('errorTEST(%d) = %d\n', i, errors_test(i));
    end
    %% update mdl
    mdl.c = c_new;
    mdl.t = t_new;
end
end