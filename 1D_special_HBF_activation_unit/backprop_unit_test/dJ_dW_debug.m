function [ dJ_dW_l ] = dJ_dW_debug( mdl, backprop, Xminibatch, fp, l , batchsize)
M = batchsize;
L = size(mdl,2);
[D_l_m_1, D_l] = size(mdl(l).W);
dJ_dW_l = zeros(D_l_m_1, D_l);
%%
if l == L && mdl(L).Act( ones(1) ) == ones(1)
    for m =1:M
        x_m = fp(l-1).A(m,:);
        dV_dW_l = zeros(D_l_m_1,D_l);
        for i = 1:D_l_m_1
            x_m_i = x_m(i);
            for j = 1:D_l
                a_i_l = x_m_i;
                dV_dW_l(i,j) = backprop(l).delta(m,j) *a_i_l;
            end
        end
        dJ_dW_l = dJ_dW_l + dV_dW_l; 
    end
    return
end
%%
for m =1:M
    if l-1 == 0
        x_m = Xminibatch(m,:);
    else      
        x_m = fp(l-1).A(m,:);
    end
    dV_dW_l = zeros(D_l_m_1,D_l);
    for i = 1:D_l_m_1
        x_m_i = x_m(i);
        for j = 1:D_l
            a_tilde_l = 2*mdl(l).beta * (x_m_i - mdl(l).W(i,j));
            dV_dW_l(i,j) = backprop(l).delta(m,j) * (a_tilde_l);
        end
    end
    dJ_dW_l = dJ_dW_l + dV_dW_l; 
end
end