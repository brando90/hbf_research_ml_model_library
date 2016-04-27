%% first delta
ds = p; % ( M x D^(L) ) = ( M x K ) % dVds = partial derivative of loss wrt scores
for j = 1:batchsize
    ds(j, Y(batch_examples(j))) = ds(j, Y(batch_examples(j))) - 1;      % (same as in logistic code) 
end
ds = ds / batchsize; % ( M x D^(L) ) = ( M x K )

prop_computation(nb_layers).xi = ds; % ( M x D^(L) )
%% 
for l = nb_layers:-1:2 
    A_l_1 = prop_computation(l-1).O; % (M x D^(l-1))
    delta_l = prop_computation(l).xi; % (M x D^(l))
    prop_computation(l).dW = A_l_1' * delta_l + lambda * neural_net(l).W; % (D ^(l-1) x M) x (M x D^(l))
    
    prop_computation(l).db = sum(delta_l, 1); %sums out the delta accross all the batch examples
    
    % compute xi for previous layer and threshold at 0 if O is <0 (ReLU gradient update)
    prop_computation(l-1).xi = delta_l * neural_net(l).W'; % (M x D^(l-1))
    prop_computation(l-1).xi(prop_computation(l-1).O <= 0) = 0;
end