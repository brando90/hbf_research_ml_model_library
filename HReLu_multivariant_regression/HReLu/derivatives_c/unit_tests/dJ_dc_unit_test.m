%% dJ_dc_unit_test
restoredefaultpath;
addpath('../..');
addpath('../../derivatives_c');
addpath('../../derivatives_t');
addpath('../../update_rules_GD');
addpath('../../model_functions');
addpath('../../analytic_tools_analysis_HBF1_GD');
addpath('../../../../common/squared_error_risk');
%%
K = 5;
D = 4;
x = rand(D,1);
x = [x; 1]
y = rand(D,1);
%% HBF1 params
c = rand(K,D);
t = rand(D+1,K);
lambda = 0;
beta = 1;
mdl = HReLu(c,t,lambda);
%%
[f_x, ~, a] = mdl.f(x);
eps = 1e-10;
%% compute dV_dc or dJ_dc
dJ_dc_loops = compute_dJ_dc_loops(f_x,y,a);
dJ_dc_numerical = compute_dV_dc_vec(f_x,a, y);
dV_dc_vec = compute_dV_dc_vec( f_x,a, y );
%% print derivatives
dJ_dc_loops
dJ_dc_numerical
dV_dc_vec