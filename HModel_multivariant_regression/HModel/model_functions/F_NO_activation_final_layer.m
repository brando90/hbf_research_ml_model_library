function [ fp ] = F_NO_activation_final_layer( mdl, Xminibatch )
%% Forward pass (fp)
batchsize = size(Xminibatch,1);
fp(0).A = Xminibatch; % (M x D+1)
fp(1).Z = [ones(batchsize,1), fp(0).A] * mdl.t;  % (M * K) = (M x D+1) x (D+1 * K)
fp(1).A = mdl.Act(0, fp(1).Z); % (M * K)
fp(2).A = fp.A * mdl.c; % (M x D) = (M * K) * (K * D)
end