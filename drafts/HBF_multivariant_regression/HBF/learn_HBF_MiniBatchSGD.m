function [ mdl, errors_train, errors_test ] = learn_HBF_MiniBatchSGD(X_train,Y_train, mdl, nb_iterations, X_test,Y_test, step_size_params, sgd_errors)
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
    A = Xminibatch;
    %% Forward Pass (fp)
    WW = sum(hbf_net(1).W .* hbf_net(1).W, 1); % ( 1 x D^(l) )
    XX = sum(A .* A, 2); % (M x 1)
    moving_offset = bsxfun(@plus, XX, WW); % ( M x D^(l) ) = (M x 1) .+ (1 x D^(l))
    Z = 2 * hbf_net.beta ( A*hbf_net(l).W - moving_offset); % (M x D^(l)) = (M x D^(l-1)+1) x (D^(l-1)+1 x D^(l))
    A = hbf_net.Act(Z); % (M x D^(l))
    fp(1).A =  A; % (M x D^(l))
    % activation for final layer
    WW = sum(hbf_net(2).W .* hbf_net(2).W, 1); % ( 1 x D^(l) )
    XX = sum(A .* A, 2); % (M x 1)
    moving_offset = bsxfun(@plus, XX, WW); % ( M x D^(l) ) = (M x 1) .+ (1 x D^(l))
    Z = 2 * hbf_net.beta ( A*hbf_net(2).W - moving_offset); % (M x D^(l)) = (M x D^(l-1)+1) x (D^(l-1)+1 x D^(l))
    A = hbf_net(2).Act(Z); % (M x D^(l))
    fp(2).A = A; % (M x D^(l))
    %% gradients * backprop
    backprop(2).delta = (2 / batchsize)*( fp(2).A - Yminibatch ) .* hbf_net(2).dAct_ds( fp(2).A ); % ( M x D^(L) ) = (M x D^(L)) .* (M x D^(L))
    step_down_1=-1;
    % get gradient matrix dV_dW^(l) for parameters W^(l) at layer l
    %%%%TODO
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