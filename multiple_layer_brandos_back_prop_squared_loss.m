%% get minibatch
mini_batch_indices = ceil(rand(batchsize,1) * nb_examples);
Xminibatch =  X(mini_batch_indices,:);
%% Forward pass starting from the input
for layer = 1:nb_layers-1
    A = max(0, out * neural_net(layer).W + repmat(neural_net(layer).b, batchsize, 1)); 
    prop_computation(layer).A = A;
end
%% Back propagation
%compute first delta = delta^(L)
delta_L = zeros(M,D_L); % ( M x D^(L) ) = ( M x K ) % dV_dz = partial derivative of loss wrt scores
%ReLu
%delta_L = (2 / batchsize)*(Yminibatch - Xminibatch) .* (prop_computation(L).A > 0);
%delta_L = (2 / batchsize)*(Yminibatch - Xminibatch);
%delta_L( prop_computation(L).O <= 0 ) = 0; % ( M x D^(L) ) = ( M x K )
%Sigmoid
%delta_L = (2 / batchsize)*(Yminibatch - Xminibatch) .* prop_computation(L).A .* (1 - prop_computation(L).A);
prop_computation(nb_layers).delta = delta_L; % ( M x D^(L) )
% compute the remaining deltas
step_down_1=-1;
for l = nb_layers:step_down_1:2
    % get gradient matrix dV_dW^(l) for parameters W^(l) at layer l
    dV_dW_l = prop_computation(l-1).A' * prop_computation(l).delta; % (M x D ^(l-1))' x (M x D^(l)) = (D ^(l-1) x M)' x (M x D^(l))
    dV_dW_l = dV_dW_l + lambda * neural_net(l).W;
    prop_computation(l).dW = dV_dW_l; % (D ^(l-1) x D^(l))
    
    %prop_computation(l).db = sum(delta_l, 1); %sums out the delta accross all the batch examples
    
    % compute delta for previous layer and threshold at 0 if O is <0 (ReLU gradient update)
    prop_computation(l-1).delta = prop_computation(l).delta * neural_net(l).W'; % (M x D^(l-1))
    prop_computation(l-1).delta(prop_computation(l-1).A <= 0) = 0;
    %prop_computation(l-1).delta = prop_computation(l).delta * neural_net(l).W' .* (prop_computation(l-1).A > 0); % (M x D^(l-1))
    % compute delta for previous layer and mutiply by derivative of Sigmoid
    prop_computation(l-1).delta = prop_computation(l).delta * neural_net(l).W' .* prop_computation(L).A .* (1 - prop_computation(L).A);
end
% gradient step for all layers
for j = 1:nb_layers
    neural_net(j).W = neural_net(j).W - step_size * prop_computation(j).dW;
    neural_net(j).b = neural_net(j).b - step_size * prop_computation(j).db;
end