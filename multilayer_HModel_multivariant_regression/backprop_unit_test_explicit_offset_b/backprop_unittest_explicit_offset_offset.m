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

sigmoid_func = @(A) sigmf(A, [1, 0]);
dSigmoid_ds = @(A) A .* (1 - A);

tanh_func = @(A) tanh(A);
dTanh_ds = @(A) 1 - A.^2;

Identity = @(A) A;
dIdentity_ds = @(A) ones(size(A));
%% dimension
N = 100;
M = 10;
D_0 = 7; % D^(L-2)
D_1 = 6; % D^(L-1)
D_2 = 5; % D^(L)
% 1 letter names
D = D_0;
K = D_1;
D_out = D_2;
%% fake data
X_train = rand(N, D);
Y_train = rand(N, D_out);
L=4;
batchsize = M;
mini_batch_indices = ceil(rand(batchsize,1) * N); % M
Xminibatch =  X_train(mini_batch_indices,:); % ( M x D ) =( M x D^(0) )
Yminibatch = Y_train(mini_batch_indices,:); % ( M x D^(L) )
%% Define multilayer net
%mdl = struct('F',cell(1,L),'Act',cell(1,L),'dAct_ds',cell(1,L),'W',cell(1,L),'beta',cell(1,L));
mdl = struct('W', cell(1,L),'b', cell(1,L),'F', cell(1,L), 'Act',cell(1,L),'dAct_ds',cell(1,L),'lambda', cell(1,L));
F_func_name = 'F_NO_activation_final_layer';
%F_func_name = 'F_activation_final_layer';
mdl(1).F = @F;
Act = sigmoid_func;
dAct_ds = dSigmoid_ds;
% Act = relu_func;
% dAct_ds = dRelu_ds;
% Act = gauss_func;
% dAct_ds = dGauss_ds;
%Act = tanh_func;
%dAct_ds = dTanh_ds;
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
D_max = D_out;
D_l_1 = D;
for l = 1:L
    if l < L
        D_l = randi(D_max,1);
        mdl(l).W = rand(D_l_1,D_l);
        mdl(l).b = rand(1,D_l);
        D_l_1 = D_l;
    else
        D_l = D_out;
        mdl(l).W = rand(D_l_1,D_l);
        mdl(l).b = rand(1,D_l);
        D_l_1 = D_l;
    end
end
%%
mdl_883 = struct('W', cell(1,L),'b', cell(1,L),'F', cell(1,L), 'Act',cell(1,L),'dAct_ds',cell(1,L),'lambda', cell(1,L));
mdl_883(1).F = @F_rep_mat;
mdl_883(1).F = @F;
for l=1:L
    mdl_883(l).W = mdl(l).W;
    mdl_883(l).b = mdl(l).b;
    mdl_883(l).Act = mdl(l).Act;
    mdl_883(l).dAct_ds = mdl(l).dAct_ds;
    mdl_883(l).lambda = 0;
end
%% Forward pass
[fp] = mdl(1).F(mdl, Xminibatch);
[fp_883] = mdl_883(1).F(mdl_883, Xminibatch);
%% Calculate derivatives using backprop code
% compute dJ_dw
backprop = struct('delta', cell(1,L));
backprop(L).delta = (2 / batchsize)*(fp(L).A - Yminibatch) .* mdl(L).dAct_ds( fp(L).A ); % ( M x D^(L) ) = (M x D^(L)) .* (M x D^(L))
step_down_1=-1;
for l = L:step_down_1:2
    backprop(l).dW = fp(l-1).A' * backprop(l).delta + mdl(l).lambda * mdl(l).W; % (D^(l-1) x D^(l)) = (M x D ^(l-1))' x (M x D^(l))
    backprop(l).db = sum(backprop(l).delta, 1); % (1 x D^(l)) = sum(M x D^(l), 1)
    % compute delta for next iteration of backprop (i.e. previous layer)
    backprop(l-1).delta = mdl(l-1).dAct_ds(fp(l-1).A) .* (backprop(l).delta * mdl(l).W'); % (M x D^(l-1)) = (M x D^(l)) .* (M x D^(l)) x ()
end
backprop(1).dW = Xminibatch' * backprop(1).delta + mdl(1).lambda * mdl(1).W; % (D^(l-1) x D^(l)) = (M x D ^(l-1))' x (M x D^(l))
backprop(1).db = sum(backprop(1).delta, 1);
%%
backprop_883 = struct('delta', cell(1,L));
backprop_883(L).delta = (2 / batchsize)*(fp_883(L).A - Yminibatch) .* mdl_883(L).dAct_ds( fp_883(L).A ); % ( M x D^(L) ) = (M x D^(L)) .* (M x D^(L))
step_down_1=-1;
for l = L:step_down_1:2
    backprop_883(l).dW = fp_883(l-1).A' * backprop_883(l).delta + mdl_883(l).lambda * mdl_883(l).W; % (D^(l-1) x D^(l)) = (M x D ^(l-1))' x (M x D^(l))
    backprop_883(l).db = sum(backprop_883(l).delta, 1); % (1 x D^(l)) = sum(M x D^(l), 1)
    % compute delta for next iteration of backprop (i.e. previous layer)
    backprop_883(l-1).delta = mdl_883(l-1).dAct_ds(fp_883(l-1).A) .* (backprop_883(l).delta * mdl_883(l).W'); % (M x D^(l-1)) = (M x D^(l)) .* (M x D^(l)) x ()
end
backprop_883(1).dW = Xminibatch' * backprop_883(1).delta + mdl_883(1).lambda * mdl_883(1).W; % (D^(l-1) x D^(l)) = (M x D ^(l-1))' x (M x D^(l))
backprop_883(1).db = sum(backprop_883(1).delta, 1);
%% Calcualte numerical derivatives
eps = 0.00000001;
numerical = struct('dW', cell(1,L),'db', cell(1,L));
numerical = numerical_derivative( numerical, eps, mdl, Xminibatch, Yminibatch);
numerical = numerical_derivative_offset( numerical, eps, mdl, Xminibatch, Yminibatch);

numerical_883 = struct('dW', cell(1,L),'db', cell(1,L));
numerical_883 = numerical_derivative( numerical_883, eps, mdl_883, Xminibatch, Yminibatch);
numerical_883 = numerical_derivative_offset( numerical_883, eps, mdl_883, Xminibatch, Yminibatch);
%% Compare with true gradient
for l = 1:L
    fprintf('------------ L = %d ------------ \n', l)
    fprintf('-- numerical(%d).dW',l);
    numerical(l).dW
    fprintf('-- numerical_883(%d).dW',l);
    numerical_883(l).dW   
    fprintf('-- backprop(%d).dW',l)
    backprop(l).dW
    fprintf('-- backprop_883(%d).dW',l)
    backprop_883(l).dW
    
    fprintf('--** \n')
    
    fprintf('numerical(%d).db',l);
    numerical(l).db
    fprintf('numerical_883(%d).db',l);
    numerical_883(l).db
    fprintf('backprop(%d).db',l)
    backprop(l).db
    fprintf('backprop_883(%d).db',l)
    backprop_883(l).db
end
beep;