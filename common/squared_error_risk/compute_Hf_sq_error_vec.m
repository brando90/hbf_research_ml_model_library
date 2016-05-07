function [ regularized_cost_hf ] = compute_Hf_sq_error_vec( X,Y, mdl)
%compute_Hf - computes regularized cost 
L = size(mdl,2);
fp = mdl(1).F(mdl, X);
F_X = fp(L).A;
regularized_penalty = 0; % TODO
N = size(X,1);
regularized_cost_hf = (1/N)*norm( F_X - Y ,'fro')^2 + mdl(1).lambda * regularized_penalty;
end