%% HMODEL
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
dIdentity_ds = @(A) ones(size(A));
%% fake data
N = 100;
D = 4;
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
Act = sigmoid_func;
dAct_ds = dSigmoid_ds;
for l = 1:L
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
K = 6;
t = rand(D+1,K);
c = rand(K+1,D_out);
mdl(1).W = t;
mdl(2).W = c;
%% Forward pass
[fp] = mdl(1).F(mdl, Xminibatch);
%% Calculate derivatives using backprop code
% compute dJ_dw
backprop = struct('delta', cell(1,L));
backprop(L).delta = (2 / batchsize)*(fp(L).A - Yminibatch) .* mdl(L).dAct_ds( fp(L).A ); % ( M x D^(L) ) = (M x D^(L)) .* (M x D^(L))
step_down_1=-1;
for l = L:step_down_1:2
    % get gradient matrix dV_dW^(l) for parameters W^(l) at layer l
    dV_dW_l = [ones(batchsize,1), fp(l-1).A]' * backprop(l).delta; % (D^(l-1)+1 x D^(l)) = (M x D ^(l-1)+1)' x (M x D^(l))
    D_l_1 = size(fp(l-1).A,1); % get value of D^(l-1)
    dV_dW_l(2:D_l_1+1,:) = dV_dW_l + mdl(l).lambda * mdl(l).W(2:D_l_1+1,:); % regularize everything except the offset
    backprop(l).dW = dV_dW_l; % (D ^(l-1)+1 x D^(l))

    % compute delta for next iteration of backprop (i.e. previous layer) and threshold at 0 if O is <0 (ReLU gradient update)
    backprop(l-1).delta = mdl.dAct_ds( fp(l-1).A ) .* backprop(l).delta * mdl(l).W'; % (M x D^(l-1)) = (M x D^(l) x ()
end
l=1;
dV_dW_l = [ones(batchsize,1), fp(l-1).A]' * backprop(l).delta; % (D^(l-1)+1 x D^(l)) = (M x D ^(l-1)+1)' x (M x D^(l))
D_l_1 = size(fp(l-1).A,1); % get value of D^(l-1)
dV_dW_l(2:D_l_1+1,:) = dV_dW_l + lambda * mdl(l).W(2:D_l_1+1,:); % regularize everything except the offset
backprop(l).dW = dV_dW_l; % (D ^(l-1)+1 x D^(l))
%% Calcualte numerical derivatives
eps = 0.001;
numerical = numerical_derivative( eps, mdl, Xminibatch, Yminibatch);
%% Compare with true gradient
for j = 1:L
    fprintf('numerical(%d).dW',j);
    numerical(j).dW
    fprintf('backprop(%d).dW',j)
    backprop(j).dW
end
beep;