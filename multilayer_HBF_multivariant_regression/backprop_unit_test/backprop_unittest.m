restoredefaultpath;clear;clc;
%% import statments
folderName = fullfile('..');
p = genpath(folderName);
addpath(p);
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
gauss_func = @(S) exp(S);
dGauss_ds = @(A) A;
Act = gauss_func;
dAct_ds = dGauss_ds;
h_mdl.Act = Act;
h_mdl.dAct_ds = dAct_ds;
h_mdl.F = @F;
K = 4;
h_mdl.beta = 0.5;
t = rand(D,K);
c = rand(K,D_out);
h_mdl(1).W = t;
h_mdl(2).W = c;
%% Forward pass
F(h_mdl, Xminibatch);
[fp] = h_mdl.F(h_mdl, Xminibatch);
%% Calculate derivatives using backprop code
% compute dJ_dw
backprop(L).delta = (2 / batchsize)*( fp(L).A - Yminibatch ) .* mdl(L).dAct_ds( fp(L).A ); % ( M x D^(L) ) = (M x D^(L)) .* (M x D^(L))
step_down_1=-1;
for l = L:step_down_1:2
    % get gradient matrix dV_dW^(l) for parameters W^(l) at layer l
    T_ijm = bsxfun( @times, W, reshape(backprop(l).delta',[1,flip( size(backprop(l).delta) )] ) ); % ( D^(l - 1) x D^(l) x M )
    backprop(l).dW = 2 * mdl.beta * ( fp(l-1).A'*backprop(l).delta - sum( T_ijm, 3) ); % (D^(l-1) x D^(l)) = (D^(l-1) x D^(l)) .- sum[ (D^(l-1) x D^(l) x M), 3 ] = (D^(l-1) x M) x (M x D^(l)) .- sum[ (D^(l-1) x D^(l) x M), 3 ]

    % compute delta for next iteration of backprop (i.e. previous layer) and threshold at 0 if O is <0 (ReLU gradient update)
    delta_sum = sum(backprop(l).delta ,2); % (M x 1) <- sum( (M x L), 2 ) 
    A_delta = bsxfun(@times, fp(l).A, delta_sum); % (M x L) = (M x L) .* (M x 1)
    backprop(l-1).delta = 2*mdl.beta * hbf_netdAct_ds( fp(l-1).A ).*( backprop(l).delta*mdl(l).W' - A_delta ); % (M x D^(l-1)) = (M x D^(l) x ()
end
%% Calcualte numerical derivatives
%TODO
numerical(2).W = 1;
numerical(1).W = 1;
%% Compare with true gradient
for j = 1:L
    numerical(j).W
    backprop(j).dW
end