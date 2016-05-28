function [ mdl, errors_train, errors_test ] = multilayer_learn_HModel_explicit_b_MiniBatchSGD( X_train,Y_train, mdl, nb_iterations,batchsize, X_test,Y_test, step, sgd_errors )
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
    if step(1).AdaGrad
        for l=1:L
            step.W(l).G_w = step.W(l).G_w + backprop(l).dW.^2;
            step.eta(l).W = step.W(l).eta ./ ( (step.W(l).G_w).^0.5 );
            step.W(l).G_b = step.W(l).G_b + backprop(l).db.^2;
            step.eta(l).b = step.b(l).eta ./ ( (step.W(l).G_b).^0.5 );
        end
    end
    % decay constant infront of step-size algorithm
    if mod(i, step.W(l).decay_frequency) == 0
        for l=1:L
            step.W(l).eta = step.W(l).eta/step.W(l).decay_rate;
        end
    end
    if mod(i, step.b(l).decay_frequency) == 0
        for l=1:L
            step.b(l).eta = step.b(l).eta/step.b(l).decay_rate;
        end
    end
    
    %% gradient step for all layers
    if step.Momentum
        for l = 1:L
            % v = a*v - eta*dJdW
            step.W(l).v =  step.W(l).alpha*step.W(l).v - step.W(l).eta .* backprop(l).dW .* mdl(l).Wmask;
            mdl(l).W = mdl(l).W + step.W(l).v;
            % v = a*v - eta*dJdb
            step.b(l).v =  step.b(l).alpha*step.b(l).v - step.b(l).eta .* backprop(l).db .* mdl(l).bmask;
            mdl(l).b = mdl(l).b + step.b(l).v;
        end
    else
        for l = 1:L
            mdl(l).W = mdl(l).W - step.W(l).eta .* backprop(l).dW .* mdl(l).Wmask;
            mdl(l).b = mdl(l).b - step.b(l).eta .* backprop(l).db .* mdl(l).bmask;
        end
    end
    %% print errors
    if sgd_errors
        errors_train(i) = compute_Hf_sq_error_vec(X_train,Y_train, mdl);
        errors_test(i) = compute_Hf_sq_error_vec(X_test,Y_test, mdl);
        %if mod(i, ceil(nb_iterations/100)) == 0 && step_size_params.print_error_to_screen
        print_every_multiple = step.print_every_multiple;
        if mod(i, print_every_multiple) == 0 && step.print_error_to_screen
            % Display the results achieved so far
            fprintf ('Iter %d. Training zero-one error: %f; Testing zero-one error: %f; step size = %f \n', i, errors_train(i), errors_test(i), step.W(l).eta)
        end
    end
end
end