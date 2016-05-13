%% HBF
restoredefaultpath;clear;clc;
%% import statments
folderName = fullfile('..');
p = genpath(folderName);
addpath(p);
folderName = fullfile('../../common/squared_error_risk');
p = genpath(folderName);
addpath(p);
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
%% Calculate derivatives using backprop code
% compute dJ_dw
backprop = struct('delta', cell(1,L));
backprop(L).delta = (2 / batchsize)*( fp(L).A - Yminibatch ) .* mdl(L).dAct_ds( fp(L).A ); % ( M x D^(L) ) = (M x D^(L)) .* (M x D^(L))
for l = L:-1:2
    if l == L && mdl(L).Act( ones(1) ) == ones(1) 
        backprop(l).dW = fp(l-1).A' * backprop(l).delta; % (D^(l-1) x D^(l)) = (M x D ^(l-1))' x (M x D^(l))
        % compute delta for next iteration of backprop (i.e. previous layer)
        backprop(l-1).delta = mdl(l-1).dAct_ds(fp(l-1).A) .* (backprop(l).delta * mdl(l).W');
    else
        % get gradient matrix dV_dW^(l) for parameters W^(l) at layer l
        T_ijm = bsxfun( @times, mdl(l).W, reshape(backprop(l).delta',[1,flip( size(backprop(l).delta) )] ) ); % ( D^(l - 1) x D^(l) x M )
        backprop(l).dW = 2 * mdl(l).beta * ( fp(l-1).A'*backprop(l).delta - sum( T_ijm, 3) ); % (D^(l-1) x D^(l)) = (D^(l-1) x D^(l)) .- sum[ (D^(l-1) x D^(l) x M), 3 ] = (D^(l-1) x M) x (M x D^(l)) .- sum[ (D^(l-1) x D^(l) x M), 3 ]
        % compute delta for next iteration of backprop (i.e. previous layer)
        delta_sum = sum(backprop(l).delta ,2); % (M x 1) <- sum( (M x D^(l)), 2 ) 
        A_x_delta = bsxfun(@times, fp(l-1).A, delta_sum); % (M x D^(L)) = (M x D^(l)) .* (M x 1)
        backprop(l-1).delta = 2*mdl(l).beta * mdl(l-1).dAct_ds( fp(l-1).A ).*( backprop(l).delta*mdl(l).W' - A_x_delta ); % (M x D^(l-1))
    end
end
l=1;
T_ijm = bsxfun( @times, mdl(l).W, reshape(backprop(l).delta',[1,flip( size(backprop(l).delta) )] ) ); % ( D^(l - 1) x D^(l) x M )
backprop(l).dW = 2 * mdl(l).beta * ( Xminibatch'*backprop(l).delta - sum( T_ijm, 3) ); % (D^(l-1) x D^(l)) = (D^(l-1) x D^(l)) .- sum[ (D^(l-1) x D^(l) x M), 3 ] = (D^(l-1) x M) x (M x D^(l)) .- sum[ (D^(l-1) x D^(l) x M), 3 ]

%% Calcualte numerical derivatives
eps = 0.0000001;
numerical = numerical_derivative_dJ_dW_l( eps, mdl, Xminibatch, Yminibatch);
%
mdl_loops = struct('F',cell(1,L),'Act',cell(1,L),'dAct_ds',cell(1,L),'W',cell(1,L),'beta',cell(1,L));
for l = 1:L
    mdl_loops(l).beta = 0.5;
    mdl_loops(l).lambda = 0;
end
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
%%
[ backprop_loops ] = dJ_dW_loops( backprop, mdl, fp, Xminibatch );
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
    
    fprintf('backprop_loops(%d).dW',l)
    backprop_loops(l).dW
    
    fprintf('dJ(%d).dW',l)
    dJ(l).dW
end
beep;