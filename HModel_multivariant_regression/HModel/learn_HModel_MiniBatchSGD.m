function [ mdl, errors_train, errors_test ] = learn_HModel_MiniBatchSGD(X_train,Y_train, mdl, nb_iterations, X_test,Y_test, step_size_params, sgd_errors)
fprintf('sgd_errors = %d',sgd_errors);
[~, K] = size(mdl.t); % (D^(0) x D^(L1))
[~, D] = size(X_train); % (M x D^(0))
[~, D_out] = size(Y_train); % (M x D^(2))
if sgd_errors
    errors_train = zeros(nb_iterations+1,1);
    errors_test = zeros(nb_iterations+1,1);
    errors_train(1) = compute_Hf_sq_error(X_train,Y_train, mdl, mdl.lambda);
    errors_test(1) = compute_Hf_sq_error(X_test,Y_test, mdl, mdl.lambda);
end
if step_size_params.AdaGrad
    G_c = ones(K, D_out);
    G_t = ones(D, K);
elseif step_size_params.Decaying
    step_size = step_size_params.step_size; %TODO
end
for i=2:length(errors_test)
    mini_batch_indices = ceil(rand(batchsize,1) * nb_examples);
    Xminibatch =  X_train(mini_batch_indices,:);
    Yminibatch = Y_train(mini_batch_indices,:);
    %% Forward Pass (fp)
%     batchsize = size(Xminibatch,1);
%     fp(0).A = Xminibatch; % (M x D+1)
%     fp(1).Z = [ones(batchsize,1), fp(0).A] * mdl.t;  % (M * K) = (M x D+1) x (D+1 * K)
%     fp(1).A = mdl.Act(0, fp(1).Z); % (M * K)
%     fp(2).A = [ones(batchsize,1), fp.A] * mdl.c; % (M x D) = (M * K+1) * (K+1 * D)
    fp = mdl.F(mdl, Xminibatch);
    %% gradients * backprop
    backprop(2).delta = (2/batchsize) * (fp(2).A - Yminibatch) .* mdl.dAct_ds(fp(2).A); % ( M x D^(L2) )
    backprop.dJ_dc = [ones(batchsize,1),fp(1).A]' * backprop(2).delta; % (D^(L1)+1 x D^(L2)) = (M x D ^(L1)+1)' x (M x D^(l))
    backprop.(1).delta = mdl.dAct_ds(fp(1).A) .* backprop(2).delta * c'; % (M x D^(L1)) = (M x D^(L1)) .* (M x D^(L2)) x (D^(L1) x D^(L2))'
    backprop.dJ_dt = fp(0).A' * backprop(1).delta; % (D^(L0)+1 x D^(L1)) = (M x D^(L0))' x (M x D^(L1))
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
    %% update mdl params
    mdl.c = mdl.c - mu_c .* (backprop.dJ_dc + mdl.lambda * mdl.c);
    mdl.t = mdl.t - mu_t .* (backprop.dJ_dt + mdl.lambda * mdl.t);
    if sgd_errors
        errors_train(i) = compute_Hf_sq_error(X_train,Y_train, mdl);
        errors_test(i) = compute_Hf_sq_error(X_test,Y_test, mdl);
    end
    if rem(i,500) == 0
        fprintf('sgd iteration = %d\n',i);
        fprintf('errorTRAIN(%d) = %d\n', i, errors_train(i));
        fprintf('errorTEST(%d) = %d\n', i, errors_test(i));
    end
end
end