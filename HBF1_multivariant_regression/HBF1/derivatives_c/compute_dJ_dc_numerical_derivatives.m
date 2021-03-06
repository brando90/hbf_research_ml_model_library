function [ dJ_dc_numerical ] = compute_dJ_dc_numerical_derivatives(x,y, c,t,beta,eps )
%compute_dJ_dc_numerical_derivative
%   Input:
%       x = data matrix (D x 1)
%       c = weights (K x L)
%       t = centers (D x K)
%       eps = epsilon (1 x 1)
%   Output:
%       dj_dc_numerical = (K, L)
[K, L] = size(c);
dJ_dc_numerical = zeros(K, L);
lambda = 0; %% TODO
for l=1:L
    for k=1:K
        e = zeros([K,L]);
        e(k,l) = eps;
        J_e1 = J_sq_error(x,y, HBF1(c + e, t, beta, lambda) );
        J_e2 = J_sq_error(x,y, HBF1(c - e, t, beta, lambda) );
        numerical_derivative = (J_e1 - J_e2)/(2*eps);
        dJ_dc_numerical(k,l) = numerical_derivative;
    end
end
end

