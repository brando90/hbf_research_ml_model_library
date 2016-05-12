function [ backprop_loops ] = dJ_dW_loops( backprop, mdl, fp, Xminibatch )
[D, K] = size(mdl(1).W);
L = size(mdl,2);
backprop_loops = struct('delta', cell(1,L), 'dW', cell(1,L));
backprop_loops(2).delta = backprop(2).delta;
backprop_loops(2).dW = backprop(2).dW;
%%
%compute delta1
for j=1:K
   backprop_loops(1).delta(j) = fp(1).A(j) * backprop_loops(2).delta * mdl(2).W(j,:)';
end
%compute dV_dt
backprop_loops(1).dW = zeros(D,K);
for i=1:D
    for j=1:K
        a_0_ij = 2 * mdl(1).beta * ( Xminibatch(i) - mdl(1).W(i,j) );
        backprop_loops(1).dW(i,j) = backprop_loops(1).delta(j)*a_0_ij;
    end
end
end