function [ fp ] = F( mdl, Xminibatch )
%% Forward Pass (fp) HBF
L = size(mdl,2);
A = Xminibatch; % ( M x D) = (M x D^(0))
fp = struct('A', cell(1,L));
c = mdl(1).W;
t = mdl(2).W;
hbf1 = HBF1(c,t,0.5,0);
for l = 1:L-1
    WW = sum(mdl(l).W.^2, 1); % ( 1 x D^(l-1)= sum( (M x D^(l)), 1 )
    XX = sum(A.^2, 2); % (M x 1) = sum( (M x D^(l-1)), 2 )
%   moving_offset = bsxfun(@plus, XX, WW); % ( M x D^(l-1) ) = (M x 1) .+ (1 x D^(l-1))
%   Z = 2*mdl(l).beta*A*mdl(l).W - mdl(l).beta * moving_offset; % (M x D^(l)) = (M x D^(l-1)) - (M x D^(l-1))
%   Z = 2*mdl(l).beta*A*mdl(l).W - mdl(l).beta * bsxfun(@plus, sum(A.^2, 2), sum(mdl(l).W.^2, 1)); % (M x D^(l)) = (M x D^(l-1)) - (M x D^(l-1))
%   Z = -mdl(l).beta*( bsxfun(@plus, sum(A.^2, 2), sum(mdl(l).W.^2, 1)) - 2*(A*mdl(l).W) );
%   Z = -mdl(l).beta*( bsxfun(@plus, WW, XX) - 2*(A*mdl(l).W) );
    Z = mdl(l).beta*( 2*(A*mdl(l).W) - bsxfun(@plus, WW, XX));
    x=Xminibatch;
    t=mdl(l).W;
    z =  -mdl(l).beta*( bsxfun(@plus, sum(t.^2)', sum(x.^2) )' - 2*(x*t) )
    Z
    fp(l).Z = Z;
    A = mdl(l).Act(Z); % (M x D^(l))
    fp(l).A = A; % (M x D^(l))
end
% activation for final layer
if mdl(L).Act( ones(1) ) == ones(1) %% if Act func is Identity i.e. NO ACTIVATION for final Layer
    A = mdl(L).Act( A * mdl(L).W ); % (M x D^(l)) = (M x D^(l-1)+1) x (D^(l-1)+1 x D^(l))
    fp(L).A = A; % (M x D^(l))
else
    WW = sum(mdl(L).W .* mdl(L).W, 1); % ( 1 x D^(l) )
    XX = sum(A .* A, 2); % (M x 1)
    moving_offset = bsxfun(@plus, XX, WW); % ( M x D^(l) ) = (M x 1) .+ (1 x D^(l))
    Z = 2 * mdl(L).beta * ( A*mdl(L).W - moving_offset); % (M x D^(l)) = (M x D^(l-1)+1) x (D^(l-1)+1 x D^(l))
    A = mdl(L).Act(Z); % (M x D^(l))
    fp(L).A = A; % (M x D^(l))
end
end