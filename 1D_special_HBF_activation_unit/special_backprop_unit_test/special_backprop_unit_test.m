%% HBF
restoredefaultpath;clear;clc;
%% import statments
folderName = fullfile('..');
p = genpath(folderName);
addpath(p);
folderName = fullfile('../../common/squared_error_risk');
p = genpath(folderName);
addpath(p);
folderName = fullfile('../../1D_special_HBF_activation_unit');
p = genpath(folderName);
addpath(p);
%% act funcs
gauss_func = @(S) exp(S);
dGauss_ds = @(A) A;

Identity = @(A) A;
dIdentity_ds = @(A) ones(1);
Act = gauss_func;
dAct_ds = dGauss_ds;
%% fake data
N = 100;
M = 11;
D = 1;
K = 9;
D_out = 4;
X_train = rand(N, D);
Y_train = rand(N, D_out);
batchsize = M;
mini_batch_indices = ceil(rand(batchsize,1) * N); % M
Xminibatch =  X_train(mini_batch_indices,:); % ( M x D ) =( M x D^(0) )
Yminibatch = Y_train(mini_batch_indices,:); % ( M x D^(L) )
%% Define multilayer HBF net
L=2;
hbf_params = struct('eps', cell(1,L) );
for l=1:L
    hbf_params(l).eps = 0.01;
end
%%
D_1 = 3;
D_2 = D_out;
hbf_params(1).W = zeros([D, D_1]);
hbf_params(2).W = zeros([D_1, D_2]);
%%
for l=1:L
    hbf_params(l).lambda = 0;
    hbf_params(l).beta = 0.5;
end
hbf_params(1).Act = Act;
hbf_params(1).dAct_ds = dAct_ds;
mdl = make_hbf( hbf_params );
mdl(1).msg = 'hbf1';

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
numerical = numerical_derivative_dJ_dW( eps, mdl, Xminibatch, Yminibatch);
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
    
    fprintf('backprop(%d).dW',l)
    backprop(l).dW
    fprintf('backprop(%d).dW2',l)
    %backprop(l).dW2
    %fprintf('dJ(%d).dW',l)
    dJ(l).dW
end
beep;