function [zero_one_loss] = evaluate_network_loss_multi (X, Y, neural_net)
    % Evalute the zero one loss for the neural network
    % Input: X is (n x d) matrix with examples as rows; Y is (n x k) multiclass or multilabel
    % with binary values 
    % neural_net is a struct with W and b for every layer of the network 
    
        nb_examples = size(X,1);
        nb_layers = size(neural_net,2);

        out = X;
        for layer = 1:nb_layers-1
            out = max(0, out * neural_net(layer).W + repmat(neural_net(layer).b, size(out,1), 1));
        end        
        scores = out * neural_net(nb_layers).W + repmat(neural_net(nb_layers).b, size(out,1), 1);

        [m, argm] = max(scores, [], 2);
        zero_one_loss = sum(argm ~= Y) / nb_examples;
    