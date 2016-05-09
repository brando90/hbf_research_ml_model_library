function [ numerical ] = numerical_derivative_offset( numerical, eps, mdl, X, Y )
L = size(mdl,2);
for l=1:L
    [~, D_l] = size(mdl(l).b);
    for d_l =1:D_l
        e = zeros([1,D_l]);
        e(d_l) = eps;
        err = compute_Hf_sq_error(X,Y, mdl);
        mdl(l).b = mdl(l).b + e;
        err_delta = compute_Hf_sq_error(X,Y, mdl);
        mdl(l).b = mdl(l).b - e;
        numerical_derivative = (err - err_delta) / eps;
        numerical(l).db(d_l) = numerical_derivative;
    end
end
end