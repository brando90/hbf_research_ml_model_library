function [ delta_l, delta_l2,delta_l3 ] = delta_l( backprop, mdl, fp, l )
[D_l, D_l_p_1] = size( mdl(l+1).W );
delta_l = zeros(1, D_l);
delta_l2 = zeros(1, D_l);
for i=1:D_l
    delta_l2(i) = 2*mdl(1).beta*mdl(l).dAct_ds(fp(l).A(i))*(backprop(l+1).delta*mdl(l+1).W(i,:)' - fp(l).A(i) .* sum(backprop(l+1).delta) );
end
%
for i=1:D_l
    total = 0;
    for j=1:D_l_p_1
        delta_l_p_1_j = backprop(l+1).delta(j);
        W_l_ij = mdl(l+1).W(i,j);
        A_l_i = fp(l).A(i);
        total = total + delta_l_p_1_j*2*mdl(1).beta*(W_l_ij - A_l_i)*mdl(l).dAct_ds(A_l_i);
        total = total + backprop(l+1).delta(j)*2*mdl(1).beta*(mdl(l+1).W(i,j) - fp(l).A(i))*mdl(l).dAct_ds(fp(l).A(i));
        %total = total + backprop(l+1).delta(j)*mdl(l+1).W(i,j) - fp(l).A * 
    end
    delta_l(i) = total;
end
%
delta_sum = sum(backprop(l).delta ,2); % (M x 1) <- sum( (M x D^(l)), 2 ) 
A_x_delta = bsxfun(@times, fp(l-1).A, delta_sum); % (M x D^(L)) = (M x D^(l)) .* (M x 1)
delta_l3 = 2*mdl(l).beta * mdl(l).dAct_ds( fp(l-1).A ).*( backprop(l).delta*mdl(l).W' - A_x_delta );
%%
delta_l
delta_l2
delta_l3
end