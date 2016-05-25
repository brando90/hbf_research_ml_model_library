function [ fp ] = F_loops_NO_activation_lots_layers( mdl, Xminibatch )
L = size(mdl,2);
[~, D_out] = size(mdl(2).W);
fp = struct('A', cell(1,L));
%% first L-1 layers
A_L_1 = Xminibatch; % (D^(0) x 1) = (1  x D^(0))'
for l=1:L-1
    [~, D_l] = size(mdl(l).W); % (D^(l-1) x D^(l))'
    Z_L = zeros(1, D_l);
    A_L = zeros(1, D_l);
    for d_l=1:D_l
       Z_L(d_l) = -mdl(l).beta * norm( A_L_1' - mdl(l).W(:,d_l), 2)^2;
       A_L(d_l) = mdl(l).Act( Z_L(d_l) );
    end
    fp(l).Z = Z_L;
    fp(l).A = A_L;
    A_L_1 = fp(l).A;
end
%% Last Layer
Z_L = zeros(1, D_l);
A_L = zeros(1, D_l);
for d_out=1:D_out
   Z_L(d_out) = fp(L-1).A * mdl(L).W(:,d_out);
   A_L(d_out) = mdl(L).Act(Z_L(d_out));
end
fp(L).Z = Z_L;
fp(L).A = A_L;
end