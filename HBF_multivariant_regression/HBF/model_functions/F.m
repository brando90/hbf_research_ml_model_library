function [ fp ] = F( mdl, Xminibatch )
%% Forward Pass (fp)
batchsize = size(Xminibatch,1);
fp(0).A = Xminibatch; % (M x D+1)
fp(1).Z = [ones(batchsize,1), fp(0).A] * mdl.t;  % (M * K) = (M x D+1) x (D+1 * K)
fp(1).A = mdl.Act(0, fp(1).Z); % (M * K)
fp(2).A = mdl(2).Act( [ones(batchsize,1), fp(1).A] * mdl.c ); % (M x D) = (M * K+1) * (K+1 * D)
end