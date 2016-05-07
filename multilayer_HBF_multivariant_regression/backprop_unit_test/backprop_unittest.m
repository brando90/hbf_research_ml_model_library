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
Act = gauss_func;
dAct_ds = dGauss_ds;
mdl(1).beta = 0.5;
mdl(1).lambda = 0;
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
t = rand(D,K);
c = rand(K,D_out);
mdl(1).W = t;
mdl(2).W = c;
%% Forward pass
[fp] = mdl(1).F(mdl, Xminibatch);
%% Calculate derivatives using backprop code
% compute dJ_dw
backprop = struct('delta', cell(1,L));
backprop(L).delta = (2 / batchsize)*( fp(L).A - Yminibatch ) .* mdl(L).dAct_ds( fp(L).A ); % ( M x D^(L) ) = (M x D^(L)) .* (M x D^(L))
step_down_1=-1;
for l = L:step_down_1:2
    % get gradient matrix dV_dW^(l) for parameters W^(l) at layer l
    T_ijm = bsxfun( @times, mdl(l).W, reshape(backprop(l).delta',[1,flip( size(backprop(l).delta) )] ) ); % ( D^(l - 1) x D^(l) x M )
    backprop(l).dW = 2 * mdl(1).beta * ( fp(l-1).A'*backprop(l).delta - sum( T_ijm, 3) ); % (D^(l-1) x D^(l)) = (D^(l-1) x D^(l)) .- sum[ (D^(l-1) x D^(l) x M), 3 ] = (D^(l-1) x M) x (M x D^(l)) .- sum[ (D^(l-1) x D^(l) x M), 3 ]

    % compute delta for next iteration of backprop (i.e. previous layer) and threshold at 0 if O is <0 (ReLU gradient update)
    delta_sum = sum(backprop(l).delta ,2); % (M x 1) <- sum( (M x D^(L)), 2 ) 
    A_delta = bsxfun(@times, fp(l-1).A, delta_sum); % (M x D^(L)) = (M x D^(L)) .* (M x 1)
    backprop(l-1).delta = 2*mdl(1).beta * mdl(l).dAct_ds( fp(l-1).A ).*( backprop(l).delta*mdl(l).W' - A_delta ); % (M x D^(l-1)) = (M x D^(l) x ()
end
%% Calcualte numerical derivatives
numerical = numerical_derivative( eps, mdl, Xminibatch, Yminibatch);
%% Compare with true gradient
for j = 1:L
    fprintf('numerical(%d).dW',j);
    numerical(j).dW
    fprintf('backprop(%d).dW',j)
    backprop(j).dW
end
beep;