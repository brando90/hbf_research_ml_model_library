%% HBF
restoredefaultpath;clear;clc;
%% import statments
folderName = fullfile('..');
p = genpath(folderName);
addpath(p);
folderName = fullfile('../../common/squared_error_risk');
p = genpath(folderName);
addpath(p);
%% act funcs
gauss_func = @(S) exp(S);
dGauss_ds = @(A) A;

relu_func = @(A) max(0,A);
dRelu_ds = @(A) A > 0;

sigmoid_func = @(A) sigmf(A, [-1, 0]);
dSigmoid_ds = @(A) A .* (1 - A);

Identity = @(A) A;
dIdentity_ds = @(A) ones(1);
%% fake data
N = 100;
D = 5;
K = 4;
D_out = 3;
X_train = rand(N, D);
Y_train = rand(N, D_out);
L=2;
batchsize = 1;
mini_batch_indices = ceil(rand(batchsize,1) * N); % M
Xminibatch =  X_train(mini_batch_indices,:); % ( M x D ) =( M x D^(0) )
Yminibatch = Y_train(mini_batch_indices,:); % ( M x D^(L) )
%% Define multilayer HBF net
%mdl = struct('F',cell(1,L),'Act',cell(1,L),'dAct_ds',cell(1,L),'W',cell(1,L),'beta',cell(1,L));
F_func_name = 'F_NO_activation_final_layer';
%F_func_name = 'F_activation_final_layer';
Act = gauss_func;
dAct_ds = dGauss_ds;
for l = 1:L
    mdl(l).beta = 0.5;
    mdl(l).lambda = 0;
end
for l =1:L-1
    mdl(l).Act = Act;
    mdl(l).dAct_ds = dAct_ds;
end
switch F_func_name
    case 'F_NO_activation_final_layer'
        mdl(L).Act = Identity;
        mdl(L).dAct_ds = dIdentity_ds;
    case 'F_activation_final_layer'
        mdl(L).Act = Act;
        mdl(L).dAct_ds = dAct_ds;
end      
mdl(1).F = @F;
t = rand(D,K);
c = rand(K,D_out);
mdl(1).W = t;
mdl(2).W = c;
%% Forward pass
[fp] = mdl(1).F(mdl, Xminibatch);
%% Calculate derivatives using backprop code
% compute dJ_dw
backprop = struct('delta', cell(1,L));
backprop(L).delta = (2 / batchsize)*( Yminibatch - fp(L).A ) .* mdl(L).dAct_ds( fp(L).A ); % ( M x D^(L) ) = (M x D^(L)) .* (M x D^(L))
backprop(L).delta2 = backprop(L).delta;
%%
[D_l, D_l_p_1] = size( mdl(l+1).W );
delta_l = zeros(1, D_l);
delta_l2 = zeros(1, D_l);
for i=1:D_l
    delta_l2(i) = 2*mdl(1).beta*mdl(l).dAct_ds(fp(l).A(i))*(backprop(l+1).delta*mdl(l+1).W(i,:)' - fp(l).A(i) .* sum(backprop(l+1).delta) );
end
%
for i=1:D_l
    total = 0;
    for j=1:D_l_p_1
        delta_l_p_1_j = backprop(l+1).delta(j);
        W_l_ij = mdl(l+1).W(i,j);
        A_l_i = fp(l).A(i);
        total = total + delta_l_p_1_j*2*mdl(1).beta*(W_l_ij - A_l_i)*mdl(l).dAct_ds(A_l_i);
        total = total + backprop(l+1).delta(j)*2*mdl(1).beta*(mdl(l+1).W(i,j) - fp(l).A(i))*mdl(l).dAct_ds(fp(l).A(i));
        %total = total + backprop(l+1).delta(j)*mdl(l+1).W(i,j) - fp(l).A * 
    end
    delta_l(i) = total;
end
%
delta_sum = sum(backprop(l).delta ,2); % (M x 1) <- sum( (M x D^(l)), 2 ) 
A_x_delta = bsxfun(@times, fp(l-1).A, delta_sum); % (M x D^(L)) = (M x D^(l)) .* (M x 1)
delta_l3 = 2*mdl(l).beta * mdl(l).dAct_ds( fp(l-1).A ).*( backprop(l).delta*mdl(l).W' - A_x_delta );
%%
delta_l
delta_l2
delta_l3