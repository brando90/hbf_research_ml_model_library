function [ mu_beta, G_beta ] = choose_step_size_beta_adagrad( eta_beta, gbeta, G_beta )
%
G_beta = G_beta + gbeta.^2;
mu_beta = eta_beta ./ ( (G_beta).^0.5 );
end