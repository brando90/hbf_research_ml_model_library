function [ fp ] = F( mdl, Xminibatch )
%% Forward Pass (fp) HBF
L = size(mdl,2);
A = Xminibatch; % ( M x D) = (M x D^(0))
fp = struct('A', cell(1,L), 'Z', cell(1,L), 'Delta_tilde', cell(1,L));
for l = 1:L
    if mod(l,2) == 1
        WW = sum(mdl(l).W.^2, 1); % ( 1 x D^(l-1)= sum( (M x D^(l)), 1 )
        XX = sum(A.^2, 2); % (M x 1) = sum( (M x D^(l-1)), 2 )
        Delta_tilde = 2*(A*mdl(l).W) - bsxfun(@plus, WW, XX) ; % 2<x,w> - (|x|^2 + |w|^2) = - (|x|^2 + |w|^2) - 2<x,w>) = -|x-w|^2
        %Z = mdl(l).beta*( 2*(A*mdl(l).W) - bsxfun(@plus, WW, XX)); % (M x D^(l-1)) - (M x D^(l-1))
        Z = mdl(l).beta*( Delta_tilde ); % (M x D^(l-1)) - (M x D^(l-1))
        A = mdl(l).Act(Z); % (M x D^(l))
        fp(l).Delta_tilde = Delta_tilde;
        fp(l).Z = Z;
        fp(l).A = A; % (M x D^(l))
    else
        A = mdl(l).Act( A * mdl(l).W ); % (M x D^(l)) = (M x D^(l-1)+1) x (D^(l-1)+1 x D^(l))
        fp(l).A = A; % (M x D^(l))
    end
end
end