gauss_func = @(S) exp(S);
dGauss_ds = @(A) A;

relu_func = @(A) max(0,A);
dRelu_ds = @(A) A > 0;

sigmoid_func = @(A) sigmf(A, [1, 0]);
dSigmoid_ds = @(A) A .* (1 - A);

tanh_func = @(A) tanh(A);
dTanh_ds = @(A) 1 - A.^2;

Identity = @(A) A;
dIdentity_ds = @(A) ones(size(A));