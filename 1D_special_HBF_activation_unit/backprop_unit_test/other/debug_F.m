restoredefaultpath;clear;clc;
%%
folderName = fullfile('../../HBF1_multivariant_regression');
p = genpath(folderName);
addpath(p);
%% HBF
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
dIdentity_ds = @(A) ones(size(A));
%% fake data
N = 100;
D = 5;
K = 4;
D_out = 3;
X_train = rand(N, D);
Y_train = rand(N, D_out);
L=2;
batchsize = 5;
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
beta = mdl(1).beta;
[fp] = mdl(1).F(mdl, Xminibatch);
hbf1 = HBF1(c,t,0.5,0);
A_L1_hbf1 = zeros(batchsize,K);
Z_hbf1 = zeros(batchsize,K);
A_L2_hbf1 = zeros(batchsize,D_out);
for m=1:batchsize
    [f_x, z, a] = hbf1.f(Xminibatch(m,:)');
    x = Xminibatch(m,:);
    t=mdl(l).W;
    %z_hard_code =  -mdl(l).beta*( bsxfun(@plus, sum(t.^2)', sum(x.^2) )' - 2*(x*t) );
    Z_hbf1(m,:) = -beta*z';
    A_L1_hbf1(m,:) = a';
    A_L2_hbf1(m,:) = f_x'; 
end
Z_hbf1
fp_Z = fp(1).Z

A_L1_hbf1
A_L1 = fp(1).A

A_L2_hbf1
A_L2 = fp(2).A