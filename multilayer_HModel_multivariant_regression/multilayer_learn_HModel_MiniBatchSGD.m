function [ mdl, errors_train, errors_test ] = multilayer_learn_HModel_MiniBatchSGD( X_train,Y_train, mdl, iterations,batchsize, X_test,Y_test, step_size_params, sgd_errors )
fprintf('sgd_errors = %d',sgd_errors);
[N, ~] = size(X_train);
[~,D_out] = size(Y_train);
L = nb_layers;
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
    step_size = step_size_params.step_size; %TODO
end
for i=2:length(errors_test)
    %% get minibatch
    mini_batch_indices = ceil(rand(batchsize,1) * N); % M
    Xminibatch =  X_train(mini_batch_indices,:); % ( M x D ) =( M x D^(0) )
    Yminibatch = Y_train(mini_batch_indices,:); % ( M x D^(L) )
    A = Xminibatch; % ( M x D+1) = (M x D^(0)+1)
    %% Forward pass starting from the input
    for l = 1:L-1
        A = mdl.Act( [ones(batchsize,1), A] * mdl(l).W ); % (M x D^(l)) = (M x D^(l-1)+1) x (D^(l-1)+1 x D^(l))
        fp(l).A = A; % (M x D^(l))
    end
    % activation for final layer
    A = mdl(L).Act( [ones(batchsize,1), A] * mdl(L).W ); % (M x D^(l)) = (M x D^(l-1)+1) x (D^(l-1)+1 x D^(l))
    fp(L).A = A; % (M x D^(l))
    %% Back propagation
    backprop(L).delta = (2 / batchsize)*(fp(L).A - Yminibatch) .* mdl(L).dAct_ds( fp(L).A ); % ( M x D^(L) ) = (M x D^(L)) .* (M x D^(L))
    step_down_1=-1;
    for l = L:step_down_1:2
        % get gradient matrix dV_dW^(l) for parameters W^(l) at layer l
        dV_dW_l = [ones(batchsize,1), fp(l-1).A]' * backprop(l).delta; % (D^(l-1)+1 x D^(l)) = (M x D ^(l-1)+1)' x (M x D^(l))
        D_l_1 = size(fp(l-1).A,1); % get value of D^(l-1)
        dV_dW_l(2:D_l_1+1,:) = dV_dW_l + lambda * mdl(l).W(2:D_l_1+1,:); % regularize everything except the offset
        backprop(l).dW = dV_dW_l; % (D ^(l-1)+1 x D^(l))

        % compute delta for next iteration of backprop (i.e. previous layer) and threshold at 0 if O is <0 (ReLU gradient update)
        backprop(l-1).delta = mdl.dAct_ds( fp(l-1).A ) .* backprop(l).delta * mdl(l).W'; % (M x D^(l-1)) = (M x D^(l) x ()
    end
    l=1;
    dV_dW_l = [ones(batchsize,1), fp(l-1).A]' * backprop(l).delta; % (D^(l-1)+1 x D^(l)) = (M x D ^(l-1)+1)' x (M x D^(l))
    D_l_1 = size(fp(l-1).A,1); % get value of D^(l-1)
    dV_dW_l(2:D_l_1+1,:) = dV_dW_l + lambda * mdl(l).W(2:D_l_1+1,:); % regularize everything except the offset
    backprop(l).dW = dV_dW_l; % (D ^(l-1)+1 x D^(l))
    %% step size
    mod_when = 2000;
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
    %% gradient step for all layers
    for j = 1:nb_layers
        mdl(j).W = mdl(j).W - step_size * backprop(j).dW;
    end
end
end