function [ delta_l1, delta_l2, delta_l3, delta_l4 ] = delta_l( backprop, mdl, fp, l, Xminibatch)
l
[D_l, D_l_p_1] = size( mdl(l+1).W );
delta_l1 = zeros(1, D_l);
delta_l2 = zeros(1, D_l);
delta_l3 = zeros(1, D_l);
delta_l4 = zeros(1, D_l);
%
for i=1:D_l
    total1 = 0;
    total2 = 0;
    for j=1:D_l_p_1
        total1 = total1 + backprop(l+1).delta(j)*2*mdl(1).beta*( mdl(l+1).W(i,j) - fp(l).A(i) ) * mdl(l).dAct_ds(fp(l).A(i));
        total2 = total2 + ( backprop(l+1).delta(j) * mdl(l+1).W(i,j) - fp(l).A(i) * backprop(l+1).delta(j));
    end
    delta_l1(i) = total1;
    delta_l2(i) = 2 * mdl(1).beta * mdl(l).dAct_ds(fp(l).A(i)) * total2;
end
%
for i=1:D_l
    delta_l3(i) = 2*mdl(1).beta*mdl(l).dAct_ds(fp(l).A(i))*(backprop(l+1).delta*mdl(l+1).W(i,:)' - fp(l).A(i) .* sum(backprop(l+1).delta) );
end
%%
l=l+1
%
d1 = size(backprop(l).delta,2);
d2 = size(fp(l-1).A,2);
delta_l4 = 2*mdl(l).beta * mdl(l-1).dAct_ds( fp(l-1).A ).*( backprop(l).delta*mdl(l).W' - fp(l-1).A .* (backprop(l).delta*ones( [d1,d2] ) ) );
%
delta_sum = sum(backprop(l).delta ,2); % (M x 1) <- sum( (M x D^(l)), 2 ) 
A_x_delta = bsxfun(@times, fp(l-1).A, delta_sum); % (M x D^(L)) = (M x D^(l)) .* (M x 1)
backprop(l-1).delta = 2*mdl(l).beta * mdl(l-1).dAct_ds( fp(l-1).A ).*( backprop(l).delta*mdl(l).W' - A_x_delta ); % (M x D^(l-1)) = (M x D^(l) x ()
delta_l5=backprop(l-1).delta;
%%
delta_l1
delta_l2
delta_l3
delta_l4
delta_l5
end