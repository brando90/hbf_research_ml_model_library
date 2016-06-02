function [ numerical ] = numerical_derivative_dJ_dW( eps, mdl, X, Y )
L = size(mdl,2);
numerical = struct('dW', cell(1,L) );
for l=1:L
    [D_l_1, D_l] = size(mdl(l).W);
    for d_l =1:D_l
        for d_l_1=1:D_l_1
            e = zeros([D_l_1,D_l]);
            e(d_l_1,d_l) = eps;
            err = compute_Hf_sq_error(X,Y, mdl);
            mdl(l).W = mdl(l).W + e;
            err_delta = compute_Hf_sq_error(X,Y, mdl);
            mdl(l).W = mdl(l).W - e;
            numerical_derivative = (err_delta - err) / eps;
            numerical(l).dW(d_l_1,d_l) = numerical_derivative;
        end
    end
end
end