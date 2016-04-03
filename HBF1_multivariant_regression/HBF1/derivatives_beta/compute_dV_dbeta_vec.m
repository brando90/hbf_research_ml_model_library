function [ dV_dbeta_vec ] = compute_dV_dbeta_vec( f,z,a, y, mdl  )
%
f_mius_y_x_ct = bsxfun(@times, 2*(f-y), mdl.c'); % (D x K) = (D x 1) op (D x K)
dV_dbeta_vec = sum(sum( bsxfun(@times, f_mius_y_x_ct , a' .* -z') ) );
end