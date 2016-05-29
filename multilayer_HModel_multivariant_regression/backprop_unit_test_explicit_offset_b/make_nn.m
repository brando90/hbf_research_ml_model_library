function [ nn ] = make_nn( nn_params)
L = size(nn_params,2);
run('./activation_funcs');
%%
nn = struct('W', cell(1,L),'b', cell(1,L),'F', cell(1,L), 'Act',cell(1,L),'dAct_ds',cell(1,L),'lambda', cell(1,L), 'beta', cell(1,L));
%activation funcs and F
Act = nn_params(1).Act;
dAct_ds = nn_params(1).dAct_ds;
for l=1:L
    if mod(l,2) == 1 % (l mod 2) = (a mod m)
        nn(l).Act = Act;
        nn(l).dAct_ds = dAct_ds;
    else % mod(l,2) == 0 even
        nn(l).Act = Identity;
        nn(l).dAct_ds = dIdentity_ds;
    end
end
%regularization
for l=1:L
    nn(l).lambda = nn_params(l).lambda;
    nn(l).beta = nn_params(l).beta;
end
%% initialize
for l=1:L
    [D_l_1, D_l] = size(nn_params(l).W);
    if mod(l,2) == 1
        nn(l).W = nn_params(l).eps * randn([D_l_1, D_l] );
        nn(l).b = nn_params(l).eps * randn([1, D_l] );
        nn(l).Wmask = 1;
        nn(l).bmask = 1;
    else
        nn(l).W = nn_params(l).eps * randn([D_l_1, D_l] );
        nn(l).b = 0;
        nn(l).Wmask = 1;
        nn(l).bmask = 0;
    end
end
nn(1).F = @F;
end