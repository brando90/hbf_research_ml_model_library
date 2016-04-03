function [ dV_dbeta_vec ] = compute_dV_dbeta_vec( f,z,a, x,y, mdl_params  )
%
f_mius_y_x_ct = bsxfun(@times, 2*(f-y), mdl_params.c'); % (D x K) = (D x 1) op (D x K)
dV_dbeta_vec = sum(sum( bsxfun(@times, f_mius_y_x_ct , a' .* -z') ) );
end