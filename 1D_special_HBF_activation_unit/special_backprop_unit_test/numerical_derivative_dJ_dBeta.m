function [ numerical ] = numerical_derivative_dJ_dBeta( eps, mdl, Xminibatch, Yminibatch)
L = size(mdl,2);
numerical = struct('dBeta', cell(1,L) );
for l=1:L
    err = compute_Hf_sq_error(Xminibatch,Yminibatch, mdl);
    mdl(l).beta = mdl(l).beta + eps;
    err_delta = compute_Hf_sq_error(Xminibatch,Yminibatch, mdl);
    mdl(l).beta = mdl(l).beta - eps;
    numerical_derivative = (err_delta - err) / eps;
    numerical(l).dBeta = numerical_derivative;
end
end