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
%% act funcs
gauss_func = @(S) exp(S);
dGauss_ds = @(A) A;

Identity = @(A) A;
dIdentity_ds = @(A) ones(1);
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
%F_func_name = 'F_activation_final_layer';
Act = gauss_func;
dAct_ds = dGauss_ds;
for l = 1:L
    mdl(l).beta = 0.5;
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
for l = L:-1:2
    if l == L && mdl(L).Act( ones(1) ) == ones(1) 
        backprop(l).dW = fp(l-1).A' * backprop(l).delta; % (D^(l-1) x D^(l)) = (M x D ^(l-1))' x (M x D^(l))
    else
        % get gradient matrix dV_dW^(l) for parameters W^(l) at layer l
        T_ijm = bsxfun( @times, mdl(l).W, reshape(backprop(l).delta',[1,flip( size(backprop(l).delta) )] ) ); % ( D^(l - 1) x D^(l) x M )
        backprop(l).dW = 2 * mdl(l).beta * ( fp(l-1).A'*backprop(l).delta - sum( T_ijm, 3) ); % (D^(l-1) x D^(l)) = (D^(l-1) x D^(l)) .- sum[ (D^(l-1) x D^(l) x M), 3 ] = (D^(l-1) x M) x (M x D^(l)) .- sum[ (D^(l-1) x D^(l) x M), 3 ]
    end
    
    % compute delta for next iteration of backprop (i.e. previous layer)
    delta_sum = sum(backprop(l).delta ,2); % (M x 1) <- sum( (M x D^(l)), 2 ) 
    A_x_delta = bsxfun(@times, fp(l-1).A, delta_sum); % (M x D^(L)) = (M x D^(l)) .* (M x 1)
    backprop(l-1).delta = 2*mdl(l).beta * mdl(l-1).dAct_ds( fp(l-1).A ).*( backprop(l).delta*mdl(l).W' - A_x_delta ); % (M x D^(l-1)) = (M x D^(l) x ()
end
l=1;
T_ijm = bsxfun( @times, mdl(l).W, reshape(backprop(l).delta',[1,flip( size(backprop(l).delta) )] ) ); % ( D^(l - 1) x D^(l) x M )
backprop(l).dW = 2 * mdl(l).beta * ( Xminibatch'*backprop(l).delta - sum( T_ijm, 3) ); % (D^(l-1) x D^(l)) = (D^(l-1) x D^(l)) .- sum[ (D^(l-1) x D^(l) x M), 3 ] = (D^(l-1) x M) x (M x D^(l)) .- sum[ (D^(l-1) x D^(l) x M), 3 ]

[ delta_l1, delta_l2, delta_l3, delta_l4 ] = delta_l( backprop, mdl, fp, 1, Xminibatch);
T_ijm = bsxfun( @times, mdl(l).W, reshape(delta_l1',[1,flip( size(delta_l1) )] ) ); % ( D^(l - 1) x D^(l) x M )
backprop(l).dW2 = 2 * mdl(l).beta * ( Xminibatch'*delta_l1 - sum( T_ijm, 3) ); % (D^(l-1) x D^(l)) = (D^(l-1) x D^(l)) .- sum[ (D^(l-1) x D^(l) x M), 3 ] = (D^(l-1) x M) x (M x D^(l)) .- sum[ (D^(l-1) x D^(l) x M), 3 ]
%%
% beta = mdl(l).beta;
% lambda = mdl(l).lambda;
% hbf1 = HBF1(c,t,beta,lambda);
% x = Xminibatch';
% y = Yminibatch';
% z =  bsxfun(@plus, sum(t.^2)', sum(x.^2) ) - 2*(t'*x); % || x - t ||^2 = ||x||^2 + ||t||^2 - 2<x,t>
% a = exp(-beta*z); %(K x 1)
% dJ_dc_vec = 2*bsxfun(@times, (f-y)', a);
% %% hbf1 numerical
% dJ_dt_numerical = compute_dJ_dt_numerical_derivatives(x,y,c,t,beta,eps);
% dJ_dc_numerical = compute_dJ_dc_numerical_derivatives(x,y,c,t,beta,eps);
%% Calcualte numerical derivatives
eps = 0.00001;
numerical = numerical_derivative_dJ_dW_l( eps, mdl, Xminibatch, Yminibatch);
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
    
%     fprintf('hbf1_numerical(%d).dW',l);
%     if l == 1
%         dJ_dt_numerical
%     else
%         dJ_dc_numerical
%     end
    
%     fprintf('hbf1_VEC_dJ(%d).dW',l);
%     if l == 1
%         dJ_dt_vec
%     else
%         dJ_dc_vec
%     end
%     
%     fprintf('hbf1_LOOPS_dJ(%d).dW',l);
%     if l == 1
%         dJ_dt_loops
%     else
%         %dJ_dc_vec
%     end
    
    fprintf('backprop(%d).dW',l)
    backprop(l).dW
    fprintf('backprop(%d).dW2',l)
    backprop(l).dW2
    fprintf('dJ(%d).dW',l)
    dJ(l).dW
end
beep;