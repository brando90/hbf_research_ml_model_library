%% dJ_dbeta_unit_test
restoredefaultpath;
addpath('..');
addpath('../..');
addpath('../../model_functions');
%% dJ_dbeta_unit_test
K = 4;
D = 3;
x = rand(D,1);
y = rand(D,1);
%% reg params
lambda = 0;
mu_c = 1;
%% make mode
c = rand(K,D);
t = rand(D,K);
beta = 0.002;
mdl = HBF1(c,t,beta,lambda);
%% Dj_dt
[f_x, z, a] = mdl.f(x);
% compute dJ_dbeta_vec
dV_dbeta_vec = compute_dV_dbeta_vec( f_x,z,a, x,y, mdl );
% compute compute_dJ_dbeta_loops
%dJ_dt_loops = compute_dJ_dbeta_loops(f_x,z,x,y,mdl.t,mdl.c);
% compute dJ_dt_numerical
eps = 1e-10;
dJ_dbeta_numerical = compute_dJ_dbeta_numerical_derivatives(x,y,mdl.c,mdl.t,mdl.beta,eps);
%% print derivatives
%dJ_dbeta_loops
dV_dbeta_vec
dJ_dbeta_numerical