%% HBF
restoredefaultpath;clear;clc;
%% import statments
folderName = fullfile('..');
p = genpath(folderName);
addpath(p);
folderName = fullfile('../../common/squared_error_risk');
p = genpath(folderName);
addpath(p);
folderName = fullfile('../../HBF1_multivariant_regression');
p = genpath(folderName);
addpath(p);
%% act funcs
gauss_func = @(S) exp(S);
dGauss_ds = @(A) A;

relu_func = @(A) max(0,A);
dRelu_ds = @(A) A > 0;

sigmoid_func = @(A) sigmf(A, [1, 0]);
dSigmoid_ds = @(A) A .* (1 - A);

Identity = @(A) A;
dIdentity_ds = @(A) ones(1);
%% fake data
N = 100;
M = 3;
D = 10;
K = 9;
D_out = 8;
X_train = rand(N, D);
Y_train = rand(N, D_out);
L=4;
batchsize = M;
mini_batch_indices = ceil(rand(batchsize,1) * N); % M
Xminibatch =  X_train(mini_batch_indices,:); % ( M x D ) =( M x D^(0) )
Yminibatch = Y_train(mini_batch_indices,:); % ( M x D^(L) )
%% Define multilayer HBF net
mdl = struct('F',cell(1,L),'Act',cell(1,L),'dAct_ds',cell(1,L),'W',cell(1,L),'beta',cell(1,L));
%F_func_name = 'F_NO_activation_final_layer';
F_func_name = 'F_activation_final_layer';
Act = gauss_func;
dAct_ds = dGauss_ds;
for l = 1:L
    mdl(l).beta = 0.12345;
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
%% Forward pass
[fp] = mdl(1).F(mdl, Xminibatch);
%% Calculate derivatives using backprop code
% compute dJ_dw
backprop = struct('delta', cell(1,L));
backprop(L).delta = (2 / batchsize)*( fp(L).A - Yminibatch ) .* mdl(L).dAct_ds( fp(L).A ); % ( M x D^(L) ) = (M x D^(L)) .* (M x D^(L))
for l = L:-1:2
    if l == L && mdl(L).Act( ones(1) ) == ones(1)
        % get gradient matrix dV_dW^(l) for parameters W^(l) at layer l
        backprop(l).dW = fp(l-1).A' * backprop(l).delta; % (D^(l-1) x D^(l)) = (M x D ^(l-1))' x (M x D^(l))
        % compute delta for next iteration of backprop (i.e. previous layer)
        backprop(l-1).delta = mdl(l-1).dAct_ds(fp(l-1).A) .* (backprop(l).delta * mdl(l).W');
    else
        % get gradient matrix dV_dW^(l) for parameters W^(l) at layer l
        T_ijm = bsxfun( @times, mdl(l).W, reshape(backprop(l).delta',[1,flip( size(backprop(l).delta) )] ) ); % ( D^(l - 1) x D^(l) x M )
        backprop(l).dW = 2 * mdl(l).beta * ( fp(l-1).A'*backprop(l).delta - sum( T_ijm, 3) ); % (D^(l-1) x D^(l)) = (D^(l-1) x D^(l)) .- sum[ (D^(l-1) x D^(l) x M), 3 ] = (D^(l-1) x M) x (M x D^(l)) .- sum[ (D^(l-1) x D^(l) x M), 3 ]
        backprop(l).dStd = -8^0.5 * mdl(l).beta^1.5 * sum( sum(backprop(l).delta .* fp(l).Delta_tilde) ); % (1 x 1)  = sum(sum( (N x D^(l)) .x (N x D^(l)) ))
        % compute delta for next iteration of backprop (i.e. previous layer)
        delta_sum = sum(backprop(l).delta ,2); % (M x 1) <- sum( (M x D^(l)), 2 ) 
        A_x_delta = bsxfun(@times, fp(l-1).A, delta_sum); % (M x D^(L)) = (M x D^(l)) .* (M x 1)
        backprop(l-1).delta = 2*mdl(l).beta * mdl(l-1).dAct_ds( fp(l-1).A ).*( backprop(l).delta*mdl(l).W' - A_x_delta ); % (M x D^(l-1))
    end
end
l=1;
T_ijm = bsxfun( @times, mdl(l).W, reshape(backprop(l).delta',[1,flip( size(backprop(l).delta) )] ) ); % ( D^(l - 1) x D^(l) x M )
backprop(l).dW = 2 * mdl(l).beta * ( Xminibatch'*backprop(l).delta - sum( T_ijm, 3) ); % (D^(l-1) x D^(l)) = (D^(l-1) x D^(l)) .- sum[ (D^(l-1) x D^(l) x M), 3 ] = (D^(l-1) x M) x (M x D^(l)) .- sum[ (D^(l-1) x D^(l) x M), 3 ]
backprop(l).dStd = -8^0.5 * mdl(l).beta^1.5 * sum( sum(backprop(l).delta .* fp(l).Delta_tilde) ); % (1 x 1)  = sum(sum( (N x D^(l)) .x (N x D^(l)) ))
%% Calcualte numerical derivatives
eps = 0.0001;
%numerical = numerical_derivative_dJ_dW( eps, mdl, Xminibatch, Yminibatch);
numerical = numerical_derivative_dJ_dstd( eps, mdl, Xminibatch, Yminibatch);
%%
% dJ = struct('dW', cell(1,L) );
% for l=1:L
%     dJ_dW_l = dJ_dW_debug( mdl, backprop, Xminibatch, fp, l, batchsize );
%     dJ(l).dW = dJ_dW_l;
% end
%dJ = struct('dBeta', cell(1,L) );
%% Compare with true gradient
fprintf('---> Derivatives \n');
for l = 1:L
    fprintf('------------------------> Derivatives layer L = %d \n', l);
%     fprintf('numerical(%d).dW',l);
%     numerical(l).dW
    
%     fprintf('backprop(%d).dW',l)
%     backprop(l).dW
%     fprintf('backprop(%d).dW2',l)
    %backprop(l).dW2
    %fprintf('dJ(%d).dW',l)
    %dJ(l).dW

    fprintf('backprop(%d).dStd',l)
    backprop(l).dStd
    fprintf('numerical(%d).dStd',l);
    numerical(l).dStd
end
beep;