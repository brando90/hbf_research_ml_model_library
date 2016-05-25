restoredefaultpath;clear;clc;clear;clc;
%%
N = 3;
D = 2;
D_out = 3;
X = rand(N,D);
%Y = rand(N,D_out);
%%%%%%%%%%%%%%%
%% make 2 hidden NN mode
run('./activation_funcs');
%Act = relu_func;
%dAct_ds = dRelu_ds;
%Act = sigmoid_func;
%dAct_ds = dSigmoid_ds;
Act = gauss_func;
dAct_ds = dGauss_ds;
lambda = 0;
%
L=3; % 3 layer, 2 hidden layer
hbf2_param = struct('Dim', cell(1,L), 'eps', cell(1,L) );
%dim of W and b
D_1 = 4;
D_2 = 2;
hbf2_param(1).Dim = [D, D_1];
hbf2_param(2).Dim = [D_1, D_2];
hbf2_param(3).Dim = [D_2, D_out];
%gaussian std/precision
std_gau_hbf2 = 0.5;
gau_precision_hbf2 = 1/(2*std_gau_hbf2);
for l=1:L
    hbf2_param(l).beta = gau_precision_hbf2;
end
%scale of init W
eps_hbf2 = 0.01;
for l=1:L
    hbf2_param(l).eps =eps_hbf2;
end
%activation funcs and F
for l=1:L-1
    hbf2_param(l).Act = Act;
    hbf2_param(l).dAct_ds = dAct_ds;
end
hbf2_param(1).F = 'F_NO_activation_final_layer';
%regularization
for l=1:L
    hbf2_param(l).lambda = 0;
end
%make NN mdl
mdl = make_HBF_model( L, hbf2_param);

%% tests
Xminibatch = X;
[ fp1 ] = F( mdl, Xminibatch )
[ fp2 ] = F_loops_NO_activation_lots_layers( mdl, Xminibatch )
%[ fp3 ] = F_loops_NO_activation( mdl, Xminibatch )
%%
fp1(L).A
fp2(L).A
% fp3(L).A