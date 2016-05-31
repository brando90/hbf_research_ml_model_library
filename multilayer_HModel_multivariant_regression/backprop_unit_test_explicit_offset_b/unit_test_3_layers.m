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
run('./activation_funcs');
%% fake data
N = 100;
D = 4;
D_out = 3;
X_train = 1+3 * rand(N, D);
Y_train = 1+6 *rand(N, D_out);
batchsize = 5;
mini_batch_indices = ceil(rand(batchsize,1) * N); % M
Xminibatch =  X_train(mini_batch_indices,:); % ( M x D ) =( M x D^(0) )
Yminibatch = Y_train(mini_batch_indices,:); % ( M x D^(L) )
%%
Act = relu_func;
dAct_ds = dRelu_ds;
%Act = sigmoid_func;
%dAct_ds = dSigmoid_ds;
%% make 3 hidden NN model
L=6;
nn_params = struct('eps', cell(1,L) );
nn_params(1).eps = 0.01;
for l=1:L
    nn_params(l).eps = 0.05;
end
%%
D_1 = 5;
D_2 = 1;
D_3 = 4;
D_4 = 1;
D_5 = 2;
D_6 = D_out;
nn_params(1).W = zeros([D, D_1]);
nn_params(2).W = zeros([D_1, D_2]);
nn_params(3).W = zeros([D_2, D_3]);
nn_params(4).W = zeros([D_3, D_4]);
nn_params(5).W = zeros([D_4, D_5]);
nn_params(6).W = zeros([D_5, D_6]);
%%
for l=1:L
    nn_params(l).lambda = 0;
    nn_params(l).beta = 0;
end
nn_params(1).Act = Act;
nn_params(1).dAct_ds = dAct_ds;
mdl = make_nn( nn_params );
mdl(1).msg = 'nn3';
%% Forward pass
[fp] = mdl(1).F(mdl, Xminibatch);
%% Calculate derivatives using backprop code
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
%% Calcualte numerical derivatives
eps = 0.00001;
numerical = struct('dW', cell(1,L),'db', cell(1,L));
numerical = numerical_derivative( numerical, eps, mdl, Xminibatch, Yminibatch);
numerical = numerical_derivative_offset( numerical, eps, mdl, Xminibatch, Yminibatch);
%% Compare with true gradient
for l = 1:L
    fprintf('------------ L = %d ------------ \n', l)
    fprintf('-- numerical(%d).dW',l);
    numerical(l).dW .* mdl(l).Wmask
    fprintf('-- backprop(%d).dW',l)
    backprop(l).dW .* mdl(l).Wmask
    
    fprintf('--** \n')
    
    fprintf('numerical(%d).db',l);
    numerical(l).db .* mdl(l).bmask
    fprintf('backprop(%d).db',l)
    backprop(l).db .* mdl(l).bmask
end
beep;