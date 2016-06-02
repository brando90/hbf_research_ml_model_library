function [ numerical ] = numerical_derivative_dJ_dstd( eps, mdl, Xminibatch, Yminibatch)
L = size(mdl,2);
numerical = struct('dStd', cell(1,L) );
for l=1:L
    err = compute_Hf_sq_error(Xminibatch,Yminibatch, mdl);
    std_new = ( 1/realsqrt(2 * mdl(l).beta) ) + eps;
    mdl(l).beta = 1/(2*std_new^2);

    err_delta = compute_Hf_sq_error(Xminibatch,Yminibatch, mdl);
    std_new = std_new - eps;
    mdl(l).beta = 1/(2*std_new^2);

    numerical_derivative = (err_delta - err) / eps;
    numerical(l).dStd = numerical_derivative;
end
end