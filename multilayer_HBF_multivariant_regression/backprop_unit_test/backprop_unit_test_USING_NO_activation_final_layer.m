%% HBF
restoredefaultpath;clear;clc;
%% import statments
folderName = fullfile('..');
p = genpath(folderName);
addpath(p);
folderName = fullfile('../../common/squared_error_risk');
p = genpath(folderName);
addpath(p);
% folderName = fullfile('../../HBF1_multivariant_regression');
% p = genpath(folderName);
% addpath(p);
%% fake data
N = 100;
M = 1;
D = 5;
K = 4;
D_out = 3;
X_train = rand(N, D);
Y_train = rand(N, D_out);
L=2;
batchsize = M;
mini_batch_indices = ceil(rand(batchsize,1) * N); % M
Xminibatch =  X_train(mini_batch_indices,:); % ( M x D ) =( M x D^(0) )
Yminibatch = Y_train(mini_batch_indices,:); % ( M x D^(L) )
%% Define multilayer-HBF net
mdl = struct('F',cell(1,L),'Act',cell(1,L),'dAct_ds',cell(1,L),'W',cell(1,L),'beta',cell(1,L));
F_func_name = 'F_NO_activation_final_layer';
mdl(L-1).Act = @(S) exp(S);
mdl(L-1).dAct_ds = @(A) A;
mdl(L).Act = @(A) A;
mdl(L).dAct_ds = @(A) ones(1);
%
%mdl(1).F = @F;
mdl(1).F = @F_NO_activation_final;
t = rand(D,K);
c = rand(K,D_out);
mdl(1).W = t;
mdl(2).W = c;
for l = 1:L
    mdl(l).beta = 0.5;
    mdl(l).lambda = 0;
end
%% Forward pass
[fp] = mdl(1).F(mdl, Xminibatch);
%% Calculate derivatives using backprop code
backprop = struct('delta', cell(1,L));
backprop(L).delta = (2 / batchsize)*( fp(L).A - Yminibatch ) .* mdl(L).dAct_ds( fp(L).A ); % ( M x D^(L) ) = (M x D^(L)) .* (M x D^(L))
backprop(L).dW = fp(L-1).A' * backprop(L).delta; % l=2 -> (D^(l-1) x D^(l)) = (M x D ^(l-1))' x (M x D^(l))
%
%delta_sum = sum(backprop(L).delta ,2); % (M x 1) <- sum( (M x D^(l)), 2 ) 
%A_x_delta = bsxfun(@times, fp(L-1).A, delta_sum); % (M x D^(L)) = (M x D^(l-1)) .* (M x 1)
%backprop(L-1).delta = 2*mdl(L-1).beta * mdl(L-1).dAct_ds( fp(L-1).A ).*( backprop(L).delta*mdl(L).W' - A_x_delta ); % (M x D^(l-1)) = (M x D^(l) x ()
backprop(L-1).delta = 2*mdl(L-1).beta * mdl(L-1).dAct_ds( fp(L-1).A ).*( backprop(L).delta*mdl(L).W' - bsxfun(@times, fp(L-1).A, sum(backprop(L).delta ,2)) );
%
T_ijm = bsxfun( @times, mdl(L-1).W, reshape(backprop(L-1).delta',[1,flip( size(backprop(L-1).delta) )] ) ); % ( D^(l - 1) x D^(l) x M )
backprop(L-1).dW = 2 * mdl(L-1).beta * ( Xminibatch'*backprop(L-1).delta - sum( T_ijm, 3) ); % (D^(l-1) x D^(l)) = (D^(l-1) x D^(l)) .- sum[ (D^(l-1) x D^(l) x M), 3 ] = (D^(l-1) x M) x (M x D^(l)) .- sum[ (D^(l-1) x D^(l) x M), 3 ]
%% Calcualte numerical derivatives
eps = 0.0000001;
numerical = numerical_derivative_dJ_dW_l( eps, mdl, Xminibatch, Yminibatch);
%
mdl_loops = struct('F',cell(1,L),'Act',cell(1,L),'dAct_ds',cell(1,L),'W',cell(1,L),'beta',cell(1,L));
F_func_name = 'F_NO_activation_final_layer';
mdl_loops(L-1).Act = @(S) exp(S);
mdl_loops(L-1).dAct_ds = @(A) A;
mdl_loops(L).Act = @(A) A;
mdl_loops(L).dAct_ds = @(A) ones(1);
%
%mdl(1).F = @F;
mdl_loops(1).F = @F_loops_NO_activation;
mdl_loops(1).W = t;
mdl_loops(2).W = c;
numerical_loops = numerical_derivative_dJ_dW_l( eps, mdl_loops, Xminibatch, Yminibatch);
%%
dJ = struct('dW', cell(1,L) );
for l=1:L
    dJ_dW_l = dJ_dW_debug( mdl, backprop, Xminibatch, fp, l, batchsize );
    dJ(l).dW = dJ_dW_l;
end
%% Compare with true gradient
fprintf('---> Derivatives \n');
for l = 1:L
    fprintf('------------------------> dJ_dw L = %d \n', l);
    fprintf('numerical(%d).dW',l);
    numerical(l).dW
    
    fprintf('numerical_loops(%d).dW',l)
    numerical_loops(l).dW
    
    fprintf('backprop(%d).dW',l)
    backprop(l).dW
    
%     fprintf('backprop(%d).dW2',l)
%     backprop(l).dW2
    
    fprintf('dJ(%d).dW',l)
    dJ(l).dW
end
beep;