function [ mdl_params ] = learn_HBF1_batch_GD(Xtrain,ytrain, mdl_params, iterations,visualize, Xtest,Ytest)
%learn_HBF_parameters_1_hidden_later - learns HBF params from Poggio's Paper
%   Inputs:
%   Outputs:
%[D, K] = size(mdl_params.t);
if visualize
    errors_Hfs = zeros(iterations,1);
    errors_Test = zeros(iterations,1);
%     changes_c = zeros(D, iterations);
%     dHf_dc_mu_c_iterion = zeros(D, iterations);
%     changes_t = zeros(K, iterations);
%     dHf_dt_mu_t_iter = zeros(K, iterations);
end
for i=1:iterations
    Kern = produce_kernel_matrix(Xtrain,mdl_params.t,mdl_params.beta ); % (N x K)
    F = Kern * mdl_params.c; % (N x D)
    %% get new parameters
    [c_new, dHf_dc, mu_c] = update_c_batch(F,Kern,ytrain, mdl_params);
    [t_new, dHf_dt, mu_t] = update_t_batch(F,Kern, Xtrain,ytrain, mdl_params);
%     if visualize
%         %% get changes for c/iter.
%         change_c_wrt_current_iteration = get_dc_diter(mdl_params.c, c_new); % (L x 1)
%         changes_c(:,i) = change_c_wrt_current_iteration; % (L x 1)
%         dJ_dc_col_norms = get_norms_col_dHf_dc(dHf_dc); % (L x 1)
%         dHf_dc_mu_c_iterion(:, i) = mu_c * dJ_dc_col_norms;
%         %% get changes for t2s/iter.
%         change_t_wrt_iteration = get_dt_diter(mdl_params.t, t_new );
%         changes_t(:, i) = change_t_wrt_iteration;
%         dHf_diter_col_norms = get_norms_col_dHf_dt(dHf_dt);
%         dHf_dt_mu_t_iter(:, i) = mu_t * dHf_diter_col_norms;
%     end
    %% update HBF1 model
    mdl_params.c = c_new;
    mdl_params.t = t_new;
    %% Calculate current errors
    if visualize
        mdl_new = HBF1(mdl_params);
        current_Hf = compute_Hf_sq_error(Xtrain,ytrain, mdl_new, mdl_params.lambda);
        current_Hf_test = compute_Hf_sq_error(Xtest,Ytest, mdl_new, mdl_params.lambda);
        errors_Hfs(i) = current_Hf;
        errors_Test(i) = current_Hf_test;
    end
end
if visualize
    %% plot error progression
    figure
    iteration_axis = 1:iterations;
    plot(iteration_axis, errors_Hfs);
%     plot(iteration_axis,errors_Hfs,'-ro',iteration_axis, errors_Test,'-b*');
%     legend('Training risk','Test risk');
    subplot(2,1,1)
    plot(iteration_axis,errors_Hfs,'-ro');
    legend('Training risk');
    subplot(2,1,2)
    plot(iteration_axis, errors_Test,'-b*');
    legend('Test risk');
    title('Train and Test risk over iteration -- ');
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
