function [ z, a ] = forward_pass_pdist2( x, t, beta )
%   Inputs:
%       x = data point (D x 1)
%       t = centers (D x K)
%       beta = precision of guasssian (1 x 1)
%   Outputs:
%       z = (K x 1)
%       a = activation the single layer (K x 1)
z = pdist2(x', t').^2'; %(1 x K)
a = exp(-beta*z); %(K x 1)
end