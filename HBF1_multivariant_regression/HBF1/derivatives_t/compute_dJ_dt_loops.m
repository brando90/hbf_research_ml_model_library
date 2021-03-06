function [ dJ_dt ] = compute_dJ_dt_loops(f_x,z,x,y,t,c)
%Computes dJ_dc
%   Input:
%       z = (K x 1)
%       x = data point (D, 1)
%       y = labels (1 x 1)
%       t = centers (D x K)
%       c = weights (K x D)
%   Output:
%       dJ_dt = (D x K)
[D,K] = size(t);
D_out =  size(c,2);
dJ_dt = zeros(D,K);
for k=1:K
    dJ_dt_k = zeros(D,1);
    for d_out=1:D_out
        c_d = c(:,d_out); %(K x 1)
        dJ_df_d = 2*(y(d_out) - f_x(d_out)); % (1 x 1)
        df_dt = -c_d(k) * exp(-z(k)) * 2 * ( x - t(:,k) ); %(D x 1)
        dJ_dt_k = dJ_dt_k + dJ_df_d * df_dt;
    end
    dJ_dt(:,k) = dJ_dt_k;
end
end