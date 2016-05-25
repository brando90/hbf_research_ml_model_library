function [ numerical ] = numerical_derivative_dJ_dBeta( eps, mdl, Xminibatch, Yminibatch)
L = size(mdl,2);
numerical = struct('dBeta', cell(1,L) );
for l=1:L
    e = zeros(l);
    e(l) = eps;
    err = compute_Hf_sq_error(Xminibatch,Yminibatch, mdl);
    mdl(l).beta = mdl(l).beta + e;
    err_delta = compute_Hf_sq_error(Xminibatch,Yminibatch, mdl);
    mdl(l).beta = mdl(l).beta - e;
    numerical_derivative = (err_delta - err) / eps;
    numerical(l).dW(l) = numerical_derivative;
end
end