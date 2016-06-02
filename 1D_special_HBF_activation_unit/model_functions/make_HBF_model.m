function [ mdl ] = make_HBF_model(L, mdl_param)
run('./activation_funcs');
mdl = struct('W', cell(1,L),'b', cell(1,L),'F', cell(1,L), 'Act',cell(1,L),'dAct_ds',cell(1,L),'lambda', cell(1,L), 'beta', cell(1,L));
%% set activation funcs
F_func_name = mdl_param(1).F;
mdl(1).F = @F;
for l=1:L
    if l==L
        switch F_func_name
        case 'F_NO_activation_final_layer'
            mdl(l).Act = Identity;
            mdl(l).dAct_ds = dIdentity_ds;
            mdl(l).beta = mdl_param(l).beta;
            mdl(l).lambda = mdl_param(l).lambda;
        case 'F_activation_final_layer'
            mdl(l).Act = mdl_param(1).Act;
            mdl(l).dAct_ds = mdl_param(1).dAct_ds;
            mdl(l).beta = mdl_param(l).beta;
            mdl(l).lambda = mdl_param(l).lambda;
        end
    else
        mdl(l).Act = mdl_param(l).Act;
        mdl(l).dAct_ds = mdl_param(l).dAct_ds;
        mdl(l).beta = mdl_param(l).beta;
        mdl(l).lambda = mdl_param(l).lambda;
    end
end
%% initialize
for l=1:L
    D_l_1 = mdl_param(l).Dim(1);
    D_l = mdl_param(l).Dim(2);
    mdl(l).W = mdl_param(l).eps * randn([D_l_1, D_l] );
end

end