function [ regularized_cost_hf ] = compute_Hf_sq_error_vec( X,Y, mdl)
%compute_Hf - computes regularized cost 
F_X = mdl.F(mdl, X);
regularized_penalty = 0; % TODO
N = size(X,1);
regularized_cost_hf = (1/N)*norm( F_X - Y ,'fro')^2 + mdl.lambda * regularized_penalty;
end