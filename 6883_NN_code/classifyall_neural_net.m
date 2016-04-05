function [] = classifyall_neural_net (x1l, x1u, x2l, x2u, delta, neural_net, k)
    % given (W,b), classifies all points in a grid [x1l,x1u]x[x2l,x2u] at
    % spacing delta into k classes
    
            [X1, X2] = meshgrid(x1l:delta:x1u, x2l:delta:x2u);
            X = [reshape(X1, size(X1,1)^2,1), reshape(X2, size(X1,1)^2,1)]; % get a matrix of 2d pairs
            
            nb_layers = size(neural_net,2);
            out =  X;
            for layer = 1:nb_layers-1
                out = max(0,out*neural_net(layer).W+repmat(neural_net(layer).b, size(out,1), 1));
                %upd(layer).out = out;
            end        
            scores = out*neural_net(nb_layers).W + repmat(neural_net(nb_layers).b, size(out,1), 1);
            
            [m, argm] = max(scores,[],2);
            cmap = colormap(hsv);
            scatter(X(:, 1), X(:, 2), 20, cmap(floor(64/k*argm),:), 'filled')
            