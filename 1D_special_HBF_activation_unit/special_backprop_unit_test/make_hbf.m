function [ hbf ] = make_hbf( hbf_params)
L = size(hbf_params,2);
run('./activation_funcs');
%%
hbf = struct('W', cell(1,L),'b', cell(1,L),'F', cell(1,L), 'Act',cell(1,L),'dAct_ds',cell(1,L),'lambda', cell(1,L), 'beta', cell(1,L));
%activation funcs and F
Act = hbf_params(1).Act;
dAct_ds = hbf_params(1).dAct_ds;
for l=1:L
    if mod(l,2) == 1 % (l mod 2) = (a mod m)
        hbf(l).Act = Act;
        hbf(l).dAct_ds = dAct_ds;
    else % mod(l,2) == 0 even
        hbf(l).Act = Identity;
        hbf(l).dAct_ds = dIdentity_ds;
    end
end
%regularization
for l=1:L
    hbf(l).lambda = hbf_params(l).lambda;
    hbf(l).beta = hbf_params(l).beta;
end
%% initialize
for l=1:L
    [D_l_1, D_l] = size(hbf_params(l).W);
    if mod(l,2) == 1
        hbf(l).W = hbf_params(l).eps * randn([D_l_1, D_l] );
        hbf(l).b = hbf_params(l).eps * randn([1, D_l] );
%         hbf(l).Wmask = 1;
%         hbf(l).bmask = 1;
    else
        hbf(l).W = hbf_params(l).eps * randn([D_l_1, D_l] );
        hbf(l).b = 0;
%         hbf(l).Wmask = 1;
%         hbf(l).bmask = 0;
    end
end
hbf(1).F = @F;
end