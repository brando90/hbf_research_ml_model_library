function [mdl, errors_train, errors_test] = multilayer_learn_HBF_MiniBatchSGD( X_train,Y_train, mdl, nb_iterations,batchsize, X_test,Y_test, step_size_params, sgd_errors )
fprintf('sgd_errors = %d \n', sgd_errors);
[N, ~] = size(X_train);
[~,D_out] = size(Y_train);
L = size(mdl,2);
if sgd_errors
    errors_train = zeros(nb_iterations+1,1);
    errors_test = zeros(nb_iterations+1,1);
    errors_train(1) = compute_Hf_sq_error(X_train,Y_train, mdl);
    errors_test(1) = compute_Hf_sq_error(X_test,Y_test, mdl);
end
i=1;
fprintf ('Iter %d. Training zero-one error: %f; Testing zero-one error: %f; eta_W =%f , eta_Std =%f \n', 0, errors_train(i), errors_test(i), step_size_params.W(1).eta, step_size_params.Std(1).eta)
%% SGD/Mini Batch
fp = struct('A', cell(1,L), 'Z', cell(1,L), 'Delta_tilde', cell(1,L));
backprop = struct('delta', cell(1,L), 'dW', cell(1,L), 'dStd', cell(1,L));
for i=2:length(errors_test)
    %% get minibatch
    mini_batch_indices = ceil(rand(batchsize,1) * N); % M
    Xminibatch =  X_train(mini_batch_indices,:); % ( M x D ) =( M x D^(0) )
    Yminibatch = Y_train(mini_batch_indices,:); % ( M x D^(L) )
    %% Forward pass starting from the input
    L = size(mdl,2);
    A = Xminibatch; % ( M x D) = (M x D^(0))
    for l = 1:L-1
        WW = sum(mdl(l).W.^2, 1); % ( 1 x D^(l-1)= sum( (M x D^(l)), 1 )
        XX = sum(A.^2, 2); % (M x 1) = sum( (M x D^(l-1)), 2 )
        Delta_tilde = 2*(A*mdl(l).W) - bsxfun(@plus, WW, XX) ;
        %Z = mdl(l).beta*( 2*(A*mdl(l).W) - bsxfun(@plus, WW, XX)); % (M x D^(l-1)) - (M x D^(l-1))
        Z = mdl(l).beta*( Delta_tilde ); % (M x D^(l-1)) - (M x D^(l-1))
        A = mdl(l).Act(Z); % (M x D^(l))
        fp(l).Delta_tilde = Delta_tilde;
        fp(l).Z = Z;
        fp(l).A = A; % (M x D^(l))
    end
    % activation for final layer
    if mdl(L).Act( ones(1) ) == ones(1) %% if Act func is Identity i.e. NO ACTIVATION for final Layer
        A = mdl(L).Act( A * mdl(L).W ); % (M x D^(l)) = (M x D^(l-1)+1) x (D^(l-1)+1 x D^(l))
        fp(L).A = A; % (M x D^(l))
    else
        WW = sum(mdl(L).W.^2, 1); % ( 1 x D^(l-1)= sum( (M x D^(l)), 1 )
        XX = sum(A.^2, 2); % (M x 1) = sum( (M x D^(l-1)), 2 )
        Delta_tilde = 2*(A*mdl(L).W) - bsxfun(@plus, WW, XX) ;
        %Z = mdl(l).beta*( 2*(A*mdl(l).W) - bsxfun(@plus, WW, XX)); % (M x D^(l-1)) - (M x D^(l-1))
        Z = mdl(L).beta*( Delta_tilde ); % (M x D^(l-1)) - (M x D^(l-1))
        A = mdl(L).Act(Z); % (M x D^(l))
        fp(L).Delta_tilde = Delta_tilde;
        fp(L).Z = Z;
        fp(L).A = A; % (M x D^(l))
    end
        
    %% Back propagation dJ_dW
    backprop(L).delta = (2 / batchsize)*( fp(L).A - Yminibatch ) .* mdl(L).dAct_ds( fp(L).A ); % ( M x D^(L) ) = (M x D^(L)) .* (M x D^(L))
    for l = L:-1:2
        if l == L && mdl(L).Act( ones(1) ) == ones(1)
            % get gradient matrix dV_dW^(l) for parameters W^(l) at layer l
            backprop(l).dW = fp(l-1).A' * backprop(l).delta; % (D^(l-1) x D^(l)) = (M x D ^(l-1))' x (M x D^(l))
            backprop(l).dStd = 0; % there is no gaussian in final layer
            % compute delta for next iteration of backprop (i.e. previous layer)
            backprop(l-1).delta = mdl(l-1).dAct_ds(fp(l-1).A) .* (backprop(l).delta * mdl(l).W');
        else
            % get gradient matrix dV_dW^(l) for parameters W^(l) at layer l
            T_ijm = bsxfun( @times, mdl(l).W, reshape(backprop(l).delta',[1,flip( size(backprop(l).delta) )] ) ); % ( D^(l - 1) x D^(l) x M )
            backprop(l).dW = 2 * mdl(l).beta * ( fp(l-1).A'*backprop(l).delta - sum( T_ijm, 3) ); % (D^(l-1) x D^(l)) = (D^(l-1) x D^(l)) .- sum[ (D^(l-1) x D^(l) x M), 3 ] = (D^(l-1) x M) x (M x D^(l)) .- sum[ (D^(l-1) x D^(l) x M), 3 ]
            backprop(l).dStd = -8^0.5 * mdl(l).beta^1.5 * sum( sum(backprop(l).delta .* fp(l).Delta_tilde) ); % (1 x 1)  = sum(sum( (N x D^(l)) .x (N x D^(l)) ))
            % compute delta for next iteration of backprop (i.e. previous layer)
            delta_sum = sum(backprop(l).delta ,2); % (M x 1) <- sum( (M x D^(l)), 2 ) 
            A_x_delta = bsxfun(@times, fp(l-1).A, delta_sum); % (M x D^(L)) = (M x D^(l)) .* (M x 1)
            backprop(l-1).delta = 2*mdl(l).beta * mdl(l-1).dAct_ds( fp(l-1).A ).*( backprop(l).delta*mdl(l).W' - A_x_delta ); % (M x D^(l-1))
        end
    end
    l=1;
    T_ijm = bsxfun( @times, mdl(l).W, reshape(backprop(l).delta',[1,flip( size(backprop(l).delta) )] ) ); % ( D^(l - 1) x D^(l) x M )
    backprop(l).dW = 2 * mdl(l).beta * ( Xminibatch'*backprop(l).delta - sum( T_ijm, 3) ); % (D^(l-1) x D^(l)) = (D^(l-1) x D^(l)) .- sum[ (D^(l-1) x D^(l) x M), 3 ] = (D^(l-1) x M) x (M x D^(l)) .- sum[ (D^(l-1) x D^(l) x M), 3 ]
    backprop(l).dStd = -8^0.5 * mdl(l).beta^1.5 * sum( sum(backprop(l).delta .* fp(l).Delta_tilde) ); % (1 x 1)  = sum(sum( (N x D^(l)) .x (N x D^(l)) ))
    %% step size
    if step_size_params(1).AdaGrad
        for l=1:L
            G_w = G_w + backprop(l).dW.^2;
            step_size_params.eta(l).W = step_size_params.eta(l).W ./ ( (G_w).^0.5 );
            G_std = G_std + backprop(l).dStd.^2;
            step_size_params.eta(l).Std = step_size_params.eta(l).Std ./ ( (G_std).^0.5 );
        end
    end
    % decay constant infront of step-size algorithm
    if mod(i, step_size_params.W(l).decay_frequency) == 0
        for l=1:L
            step_size_params.W(l).eta = step_size_params.W(l).eta/step_size_params.W(l).decay_rate;
        end
    end
    if mod(i, step_size_params.Std(l).decay_frequency) == 0
        for l=1:L
            step_size_params.Std(l).eta = step_size_params.Std(l).eta/step_size_params.Std(l).decay_rate;
        end
    end
    %% gradient step for all layers
    if step(1).Momentum
        for l = 1:L
            % v = a*v - eta*dJdW
            step.W(l).v =  step(l).alpha*step.W(l).v - step.W(l).eta .* backprop(l).dW .* mdl(l).Wmask;
            mdl(l).W = mdl(l).W + step.W(l).v;
            % v = a*v - eta*dJdstd
            step.std(l).v =  step(l).alpha*step.std(l).v - step.std(l).eta .* backprop(l).std .* mdl(l).stdmask;
            std_new = ( 1/realsqrt(2 * mdl(l).beta) ) + step.std(l).v;
            mdl(l).std = 1/(2*std_new^2);
        end
    else
        for l = 1:L
            % W = W - eta dJdW
            mdl(l).W = mdl(l).W - step_size_params.W(l).eta * backprop(l).dW;
            % std = std - eta dJdstd
            std_new = ( 1/realsqrt(2 * mdl(l).beta) ) - step_size_params.Std(l).eta * backprop(l).dStd;
            mdl(l).beta = 1/(2*std_new^2);
        end
    end
    %% print errors
    if sgd_errors
        errors_train(i) = compute_Hf_sq_error_vec(X_train,Y_train, mdl);
        errors_test(i) = compute_Hf_sq_error_vec(X_test,Y_test, mdl);
        %if mod(i, ceil(nb_iterations/100)) == 0 && step_size_params.print_error_to_screen
        print_every_multiple = step_size_params.print_every_multiple;
        if mod(i, print_every_multiple) == 0 && step_size_params.print_error_to_screen
            % Display the results achieved so far
            fprintf ('%s: Iter %d. Training zero-one error: %f; Testing zero-one error: %f; eta_W =%f , eta_Std =%f \n', mdl(1).msg, i, errors_train(i), errors_test(i), step_size_params.W(1).eta, step_size_params.Std(1).eta)
        end
    end
end
end