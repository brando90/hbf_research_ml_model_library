function [ fp ] = F_loops_NO_activation( mdl, Xminibatch )
L = size(mdl,2);
[~, K] = size(mdl(1).W);
[~, D_out] = size(mdl(2).W);
fp = struct('A', {zeros(1,K),zeros(1,D_out)}, 'Z', {zeros(1,K),zeros(1,D_out)} );
%% First Layer
A_0 = Xminibatch';
for k=1:K
   fp(1).Z(k) = -mdl(1).beta * norm( A_0 - mdl(1).W(:,k), 2)^2;
   fp(1).A(k) = mdl(1).Act(fp(1).Z(k));
end
%% Second Layer
for d_out=1:D_out
   fp(2).Z(d_out) = fp(1).A * mdl(2).W(:,d_out);
   fp(2).A(d_out) = mdl(2).Act(fp(2).Z(d_out));
end
end