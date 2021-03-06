function [ dJ_dh ] = compute_dJ_dh(h,y,K)
%computes 1{y==l} - p(y=l|x;theta)
%   Input:
%       h = h's (L x 1)
%       y = labels (1 x 1)
%   Output:
%       dJ_df = dJ_df (K x L) 
L = length(h);
ind_y_l = ((1:L) == y); %(1 x L)
ind_y_l_M = repmat(ind_y_l, K, 1); %(K2 x L)
prob = prob_y_x(h); %(L x 1)
prob_M = repmat(prob', K, 1); %(K2 x L)
delta = ind_y_l_M - prob_M;
dJ_dh = delta;
end