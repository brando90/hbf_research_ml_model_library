function [ fp ] = F( mdl, Xminibatch )
%% Forward Pass (fp) HModel/NN
L = size(mdl,2);
A = Xminibatch; % ( M x D) = (M x D^(0))
batchsize = size(Xminibatch,1);
for l = 1:L-1
    A = mdl(l).Act( [ones(batchsize,1), A] * mdl(l).W ); % (M x D^(l)) = (M x D^(l-1)+1) x (D^(l-1)+1 x D^(l))
    fp(l).A = A; % (M x D^(l))
end
% activation for final layer
A = mdl(L).Act( [ones(batchsize,1), A] * mdl(L).W ); % (M x D^(l)) = (M x D^(l-1)+1) x (D^(l-1)+1 x D^(l))
fp(L).A = A; % (M x D^(l))
end