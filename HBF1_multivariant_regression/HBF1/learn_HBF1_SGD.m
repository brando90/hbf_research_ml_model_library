function [ mdl, errors_train, errors_test ] = learn_HBF1_SGD(X_train,Y_train, mdl, iterations,visualize, X_test,Y_test, eta_c,eta_t,eta_beta, sgd_errors)
%learn_HBF_parameters_1_hidden_later - learns HBF params from Poggio's Paper
fprintf('sgd_errors = %d \n',sgd_errors);
fprintf('visualize = %d \n',visualize);
[~, K] = size(mdl.t);
[D, N] = size(X_train);
[D_out, ~] = size(Y_train);
if visualize || sgd_errors
    errors_train = zeros(iterations+1,1);
    errors_test = zeros(iterations+1,1);
%     changes_c = zeros(D, iterations);
%     dHf_dc_mu_c_iterion = zeros(D, iterations);
%     changes_t = zeros(K, iterations);
%     dHf_dt_mu_t_iter = zeros(K, iterations);
end
if eta_c ~= 0
    G_c = ones(K, D_out);
end
if eta_t ~= 0
    G_t = ones(D, K);
end
if eta_beta ~= 0
    G_beta = ones(1, 1);
end
%% compute the error of the model without having moved the centers yet
if visualize || sgd_errors
    current_train_error = compute_Hf_sq_error(X_train,Y_train, mdl, mdl.lambda);
    current_error_test = compute_Hf_sq_error(X_test,Y_test, mdl, mdl.lambda);
    errors_train(1) = current_train_error;
    errors_test(1) = current_error_test;
end
for i=2:length(errors_test)
    if rem(i,500) == 0
        fprintf('sgd iteration = %d\n',i);
    end
    %% choose random data point x,y
    i_rand = randi(N);
    x = X_train(:,i_rand);
    y = Y_train(:,i_rand);
    %% get new parameters
    [ f_x, z, a ] = mdl.f(x);
    if eta_beta ~= 0
        [beta_new, dV_dbeta,G_beta, mu_beta] = update_beta_stochastic(f_x,z,a, y, mdl, G_beta,eta_beta);
        %change_in_beta = mu_beta .* dV_dbeta
    end
    if eta_c ~= 0
        [c_new, dV_dc,G_c, mu_c] = update_c_stochastic(f_x,a, x,y, mdl, G_c,eta_c);
        %average_change_in_c = mean(mean(dV_dc .* G_c))
    end
    if eta_t ~= 0
        [t_new, dV_dt,G_t, mu_t] = update_t_stochastic(f_x,a, x,y, mdl, G_t,eta_t);
        %average_change_in_t = mean(mean(dV_dt .* G_t))
    end
%     %% get changes for c/iter.
%     change_c_wrt_current_iteration = get_dc_diter(mdl_new.c, c_new); % (L x 1)
%     changes_c(:,i) = change_c_wrt_current_iteration; % (L x 1)
%     dJ_dc_col_norms = get_norms_col_dHf_dc(dJ_dc); % (L x 1)
%     dHf_dc_mu_c_iterion(:, i) = mu_c * dJ_dc_col_norms;
%     %% get changes for t2s/iter.
%     change_t_wrt_iteration = get_dt_diter(mdl_new.t, t_new );
%     changes_t(:, i) = change_t_wrt_iteration;
%     dHf_diter_col_norms = get_norms_col_dHf_dt(dJ_dt);
%     dHf_dt_mu_t_iter(:, i) = mu_t * dHf_diter_col_norms;
    %% update HBF1 model
    if eta_c ~= 0
        mdl.c = c_new; 
    end
    if eta_t ~= 0
        mdl.t = t_new; 
    end
    if eta_beta ~= 0
       mdl.beta = beta_new; 
    end
    %% Calculate current errors
    if visualize || sgd_errors
        current_train_error = compute_Hf_sq_error(X_train,Y_train, mdl, mdl.lambda);
        current_error_test = compute_Hf_sq_error(X_test,Y_test, mdl, mdl.lambda);
        errors_train(i) = current_train_error;
        errors_test(i) = current_error_test;
    end
end
if visualize
    %% plot error progression
    figure
    %iterations = length(errors_test);
    iterations = length(errors_test);
    iteration_axis = 0:iterations-1;
    %iteration_axis = 0:iterations;
    plot(iteration_axis,errors_train,'-ro',iteration_axis, errors_test,'-b*');
    %plot(iteration_axis(4100:iterations) ,errors_Hfs(4100:iterations) ,'-ro',iteration_axis(4100:iterations) , errors_Test(4100:iterations) ,'-b*');
    legend('Training risk','Test risk');
    xlabel('SGD iterations') % x-axis label
    ylabel('(Squared) Error') % y-axis label
    title(sprintf('Train and Test risk over iteration -- (eta_c , e_t) = ( %d , %d )', eta_c, eta_t) );
    %% plot changes in param c
%     D = min(D,50);
%     for l=1:D
%         figure
%         c_changes_l = changes_c(l,:); % (1 x iterations)
%         subplot(2,1,1)
%         plot(iteration_axis, c_changes_l)
%         title(strcat('delta c/iter -- ', num2str(l)))
% 
%         dHf_dc_mu_c = dHf_dc_mu_c_iterion(l,:); % (1 x iterations)
%         subplot(2,1,2)
%         plot(iteration_axis, dHf_dc_mu_c)
%         title(strcat('dJ dc -- ', num2str(l)))
%     end
%     %% plot changes in param t
%     K = min(K,50);
%     for k=1:K
%         figure
%         t_changes_k = changes_t(k,:); % (1 x iterations)
%         subplot(2,1,1)
%         plot(iteration_axis, t_changes_k)
%         title(strcat('t/iter -- ', num2str(k)))
% 
%         dHf_dt_mu_t = dHf_dt_mu_t_iter(k,:); % (1 x iterations)
%         subplot(2,1,2)
%         plot(iteration_axis, dHf_dt_mu_t)
%         title(strcat('dJ dt -- ', num2str(k)))
%     end
end
end
