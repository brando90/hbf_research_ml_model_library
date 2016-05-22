function [ mdl, errors_train, errors_test ] = multilayer_learn_HModel_explicit_b_MiniBatchSGD( X_train,Y_train, mdl, nb_iterations,batchsize, X_test,Y_test, step_size_params, sgd_errors )
fprintf('sgd_errors = %d',sgd_errors);
[N, ~] = size(X_train);
[~,D_out] = size(Y_train);
L = size(mdl,2);
if sgd_errors
    errors_train = zeros(nb_iterations+1,1);
    errors_test = zeros(nb_iterations+1,1);
    errors_train(1) = compute_Hf_sq_error_vec(X_train,Y_train, mdl);
    errors_test(1) = compute_Hf_sq_error_vec(X_test,Y_test, mdl);
end
if step_size_params.AdaGrad
    G_c = ones(K, D_out);
    G_t = ones(D, K);
elseif step_size_params.Decaying
    step_size = step_size_params.step_size; %TODO
end
fp = struct('A', cell(1,L));
backprop = struct('delta', cell(1,L), 'dW', cell(1,L), 'db', cell(1,L) );
for i=2:length(errors_test)
    %% get minibatch
    mini_batch_indices = ceil(rand(batchsize,1) * N);
    Xminibatch =  X_train(mini_batch_indices,:); % ( M x D ) =( M x D^(0) )
    Yminibatch = Y_train(mini_batch_indices,:); % ( M x D^(L) )
    %% Forward pass starting from the input
    A = Xminibatch; % ( M x D+1) = (M x D^(0))
    for l = 1:L-1
        A = mdl(l).Act( bsxfun(@plus, A * mdl(l).W, mdl(l).b) ); % (M x D^(l)) = (M x D^(l-1)) x (D^(l-1) x D^(l)) .+ (1 x D^(l))
        %A = max(0, A * mdl(l).W + repmat(mdl(l).b, batchsize, 1));
        fp(l).A = A; % (M x D^(l))
    end
    % activation for final layer (not special for regression but special for classification as we need to output probability of each class
    A = mdl(L).Act( bsxfun(@plus, A * mdl(L).W, mdl(L).b) ); % (M x D^(l)) = (M x D^(l-1)) x (D^(l-1) x D^(l)) .+ (1 x D^(l))
    fp(L).A = A; % (M x D^(l))
    %% Back propagation dJ_dW
    backprop(L).delta = (2 / batchsize)*(fp(L).A - Yminibatch) .* mdl(L).dAct_ds( fp(L).A ); % ( M x D^(L) ) = (M x D^(L)) .* (M x D^(L))
    step_down_1=-1;
    for l = L:step_down_1:2
        backprop(l).dW = fp(l-1).A' * backprop(l).delta + mdl(l).lambda * mdl(l).W; % (D^(l-1) x D^(l)) = (M x D ^(l-1))' x (M x D^(l))
        backprop(l).db = sum(backprop(l).delta, 1); % (1 x D^(l)) = sum(M x D^(l), 1)
        % compute delta for next iteration of backprop (i.e. previous layer)
        backprop(l-1).delta = mdl(l-1).dAct_ds(fp(l-1).A) .* (backprop(l).delta * mdl(l).W'); % (M x D^(l-1)) = (M x D^(l)) .* (M x D^(l)) x ()
    end
    backprop(1).dW = Xminibatch' * backprop(1).delta + mdl(1).lambda * mdl(1).W; % (D^(l-1) x D^(l)) = (M x D ^(l-1))' x (M x D^(l))
    backprop(1).db = sum(backprop(1).delta, 1);
    %% step-size
    % TODO how to process step-size for offset
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
        mdl(j).b = mdl(j).b - step_size * backprop(j).db;
    end
    %% print errors
    if sgd_errors
        errors_train(i) = compute_Hf_sq_error_vec(X_train,Y_train, mdl);
        errors_test(i) = compute_Hf_sq_error_vec(X_test,Y_test, mdl);
        %if mod(i, ceil(nb_iterations/100)) == 0 && step_size_params.print_error_to_screen
        if mod(i, ceil(nb_iterations/100)) == 0 && step_size_params.print_error_to_screen
            % Display the results achieved so far
            fprintf ('Iter %d. Training zero-one error: %f; Testing zero-one error: %f; step size = %f \n', i, errors_train(i), errors_test(i), step_size)
        end
    end
end
end