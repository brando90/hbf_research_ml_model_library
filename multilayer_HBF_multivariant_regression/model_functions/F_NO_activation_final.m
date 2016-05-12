function [ fp ] = F_NO_activation_final( mdl, Xminibatch )
%% Forward Pass (fp) HBF
L = size(mdl,2);
A = Xminibatch; % ( M x D) = (M x D^(0))
fp = struct('A', cell(1,L));
for l = 1:L-1
    WW = sum(mdl(l).W.^2, 1); % ( 1 x D^(l-1)= sum( (M x D^(l)), 1 )
    XX = sum(A.^2, 2); % (M x 1) = sum( (M x D^(l-1)), 2 )
    Z = mdl(l).beta*( 2*(A*mdl(l).W) - bsxfun(@plus, WW, XX)); % (M x D^(l-1)) - (M x D^(l-1))
    fp(l).Z = Z;
    A = mdl(l).Act(Z); % (M x D^(l))
    fp(l).A = A; % (M x D^(l))
end
% activation for final layer
A = mdl(L).Act( A * mdl(L).W ); % (M x D^(l)) = (M x D^(l-1)+1) x (D^(l-1)+1 x D^(l))
fp(L).A = A; % (M x D^(l))
end