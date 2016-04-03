function [ beta_new, dJ_dbeta,G_beta_new, mu_beta] = update_beta_stochastic(f_x,z,a, y, mdl, G_beta,eta_beta)
%HBF1
dJ_dbeta = compute_dV_dbeta_vec( f_x,z,a, y, mdl ); % f,a, y
[mu_beta, G_beta_new] = choose_step_size_c_adagrad(eta_beta, dJ_dbeta, G_beta);
%mu_c = choose_step_size_c_stochastic_closed_soln(f_x, a, dJ_dc, y);
%mu_c = choose_step_size_c_stochastic(x,y, mdl_params.c,mdl_params.t,mdl_params.beta, dJ_dc );
dJ_dbeta = dJ_dbeta + mdl.lambda * 0; %TODO
%% update
%c_new = mdl_params.c - mu_c * dJ_dc;
beta_new = mdl.beta - mu_beta .* dJ_dbeta;
end