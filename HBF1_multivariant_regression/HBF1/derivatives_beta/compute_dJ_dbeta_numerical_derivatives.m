function [ dJ_dt_numerical ] = compute_dJ_dbeta_numerical_derivatives(x,y,c,t,beta,eps)
%compute_dJ_dt_numerical_derivative
%   Input:
%       x = data matrix (D x 1)
%       c = weights (K x L)
%       t = centers (D x K)
%       beta = precision (1 x 1)
%       eps = epsilon (1 x 1)
%   Output:
%       dJ_dt_numerical = (D, K)
lambda = 0; %TODO
J_e2 = norm( HBF1(c,t, beta+eps, lambda).predict(x) - y, 2)^2;
J_e1 = norm( HBF1(c,t, beta-eps, lambda).predict(x) - y, 2)^2;
numerical_derivative = (J_e2 - J_e1)/(2*eps);
dJ_dt_numerical = numerical_derivative;
end