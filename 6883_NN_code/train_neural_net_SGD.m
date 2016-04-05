% Written by A. Rakhlin and A. Flajolet, 2016
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Trains a multi-layer NN for multiclass classification. Training is done via SGD/minibatch/ERM. 
% Loss is softmax, the coordinate-wise non-linear mapping used is ReLU except for the last layer

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% X is a matrix (n, d) with n the number of training examples and d the dimension of the training vectors
% Y is a matrix (n, 1) with class labels for each datum (numbered 1:k)
% Xtest and Ytest are test data to display the out-of-sample performance
% neural_net is a "struct" with fields neural_net(layer).W and neural_net(layer).b for any number of layers
% k = number of classes
% step_size = parameter for gradient descent
% lambda = regularization of weights in logistic regression. enters in weight update
% num_iter = number of gradient updates
% batchsize = size of minibatching if ERM == 0
% ERM == 1 means do full gradient, 0 means do SGD with batchsize
% visualize = 1 if want to visualize 2d solution

function [neural_net] = train_neural_net_SGD_Nhidden (X, Y, Xtest, Ytest, neural_net, k, step_size, lambda, num_iter, batchsize, ERM, visualize)

    nb_examples = size(X,1);
    nb_layers = size(neural_net,2);
    cmap = colormap(hsv);
    perf_train = [];
    perf_test = [];
    
    for i = 1:num_iter
        
        if ERM == 1             % if full gradient
            %disp('ERM')
            batch_examples = [1:nb_examples]'; 
            batchsize = nb_examples;
        else                    % else SGD with minibatch size
            % take a random subsample of size batchsize
            %disp('SGD')
            batch_examples = ceil(rand(batchsize,1) * nb_examples);
        end
        
        %%%%%%% forward pass starting from the input
        out =  X(batch_examples,:); 
        for layer = 1:nb_layers-1
            out = max(0, out * neural_net(layer).W + repmat(neural_net(layer).b, batchsize, 1)); 
            prop_computation(layer).O = out;
        end
        % last layer, compute the class probabilities p
        scores = out * neural_net(nb_layers).W + repmat(neural_net(nb_layers).b, batchsize, 1);
        scores = scores - max(scores, [], 2) * ones(1, k); % subtract off constant per row for numerical stabiltiy
        expscores = exp(scores);
        p = expscores ./ repmat(sum(expscores,2), 1, k);
        %%%%%%%
        
        if mod(i, ceil(num_iter/100)) == 0
                % Display the results achieved so far
                [ztrain] = evaluate_network_loss (X, Y, neural_net);
                [ztest] = evaluate_network_loss (Xtest, Ytest, neural_net);
                perf_train = [perf_train ztrain]; perf_test = [perf_test ztest];
                fprintf ('Iter %d. Training zero-one error: %f; Testing zero-one error: %f; step size =%f \n', i, ztrain, ztest, step_size)
                if visualize
                    subplot(1,2,1); classifyall_neural_net(-2,2,-2,2,0.04,neural_net,k); drawnow; 
                    subplot(1,2,2); scatter(X(:, 1), X(:, 2), 10, cmap(floor(64/k*Y),:), 'filled'); drawnow;
                end
        end

        
        %%%%%%% backward pass starting from the output, i.e. the class probabilities
        ds = p;                                                                 % partial derivative of loss wrt scores
        for j = 1:batchsize
            ds(j, Y(batch_examples(j))) = ds(j, Y(batch_examples(j))) - 1;      % (same as in logistic code) 
        end
        ds = ds / batchsize;

        prop_computation(nb_layers).xi = ds;
        
        for j = nb_layers:-1:2     % compare this code to the updates provided in lecture
            prop_computation(j).dW = prop_computation(j-1).O' * prop_computation(j).xi + lambda * neural_net(j).W; %shrink weights
            prop_computation(j).db = sum(prop_computation(j).xi, 1);
            % compute xi for previous layer and threshold at 0 if O is <0 (ReLU gradient update)
            prop_computation(j-1).xi = prop_computation(j).xi * neural_net(j).W';
            prop_computation(j-1).xi(prop_computation(j-1).O <= 0) = 0;
        end
        
        prop_computation(1).dW = X(batch_examples,:)' * prop_computation(1).xi + lambda * neural_net(1).W; %shrink weights
        prop_computation(1).db = sum(prop_computation(1).xi, 1);
        %%%%%%%
        
        
        %%%%%%% subgradient updates
        % update step size (e.g. decrease every ... iterations)
%         mod_when = 2000;
%         if 10000 < i
%             if mod(i, mod_when) == 0
%                 step_size = step_size/(1 * sqrt(i));
%             end
%         end
        mod_when = 2000;
        if mod(i, mod_when) == 0
            step_size = step_size/1.2;
            %step_size = step_size/sqrt(i);
            %step_size = step_size/(1 * log(i));
            %step_size = step_size/0.1;
        end
        %%%ada_grad
        %G_beta = G_beta + gbeta.^2;
        %mu_beta = eta_beta ./ ( (G_beta).^0.5 );
        
        % gradient step for all layers
        for j = 1:nb_layers
            neural_net(j).W = neural_net(j).W - step_size * prop_computation(j).dW;
            neural_net(j).b = neural_net(j).b - step_size * prop_computation(j).db;
        end
        
        %%%%%%% 
        
    end
    
    min_train = min(perf_train)
    min_test = min(perf_test)
    figure;
    plot(perf_train, 'r'); hold on;
    plot(perf_test, 'b'); ylabel('Training/testing error');
