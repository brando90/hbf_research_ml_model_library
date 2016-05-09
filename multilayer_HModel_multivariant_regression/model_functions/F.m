function [ fp ] = F( mdl, Xminibatch )
%% Forward Pass (fp) HModel/NN
L = size(mdl,2);
A = Xminibatch; % ( M x D) = (M x D^(0))
batchsize = size(Xminibatch,1);
fp = struct('A', cell(1,L));
if isfield(mdl, 'b')
    A = Xminibatch; % ( M x D+1) = (M x D^(0))
    for l = 1:L-1
        A = mdl(l).Act( bsxfun(@plus, A * mdl(l).W, mdl(l).b) ); % (M x D^(l)) = (M x D^(l-1)) x (D^(l-1) x D^(l)) .+ (1 x D^(l))
        fp(l).A = A; % (M x D^(l))
    end
    % activation for final layer (not special for regression but special for classification as we need to output probability of each class
    A = mdl(L).Act( bsxfun(@plus, A * mdl(L).W, mdl(L).b) ); % (M x D^(l)) = (M x D^(l-1)) x (D^(l-1) x D^(l)) .+ (1 x D^(l))
    fp(L).A = A; % (M x D^(l))
else
    A = Xminibatch; % ( M x D+1) = (M x D^(0))
    for l = 1:L-1
        A = mdl(l).Act( [ones(batchsize,1), A] * mdl(l).W ); % (M x D^(l)) = (M x D^(l-1)+1) x (D^(l-1)+1 x D^(l))
        fp(l).A = A; % (M x D^(l))
    end
    % activation for final layer
    A = mdl(L).Act( [ones(batchsize,1), A] * mdl(L).W ); % (M x D^(l)) = (M x D^(l-1)+1) x (D^(l-1)+1 x D^(l))
    fp(L).A = A; % (M x D^(l))
end;
end