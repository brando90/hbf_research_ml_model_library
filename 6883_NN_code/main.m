clear;
addpath ../problem_1/;

nb_iter = 50000;

%load spirals data
%d = 2; k= 5; n = 50; curvature = 2; jitter = 0.2;
%d = 2; k= 5; n = 100; curvature = 3; jitter = 0.3;
%d = 2; k= 5; n = 100; curvature = 5; jitter = 0.2;
%[X,Y] = generate_spirals (n, k, curvature, jitter, 0, 0);
%%%%FILL IN: Data normalization code
%mu = mean(X,1);
%stdev = std(X,0,1);
%X = X - repmat(mu,n*k,1);
%X = X ./ repmat(stdev,n*k,1);


load('mnist_dirty_data.mat');


%%%%
% cmap = colormap(hsv);
% scatter(X(:, 1), X(:, 2), 10, cmap(floor(64/k*Y),:), 'filled')

% Initialize parameters
clear neural_net;
W_scale = 0.1; % scale for random W entries
k = 10;
d = size(Xtrain_normal,2)
%%
r1 = 100
r2 = 100
r3 = 100 
r4 = 100   % sizes for layers of NN
%r1 = 30; r2 = 30; r3 = 30; % r4 = 20;
%%
neural_net(1).W = W_scale * randn(d,r1); % first layer is d x r1
neural_net(1).b = zeros(1,r1);   
%neural_net(1).W = W_scale * randn(d,k); % first layer is d x r1
%neural_net(1).b = zeros(1,k); 

%neural_net(2).W = W_scale * randn(r1,k); % make last layer's dimension k
%neural_net(2).b = zeros(1,k); 
neural_net(2).W = W_scale * randn(r1,r2); % make last layer's dimension k
neural_net(2).b = zeros(1,r2); 

%neural_net(3).W = W_scale * randn(r2, k);
%neural_net(3).b = zeros(1,k);
neural_net(3).W = W_scale * randn(r2, r3);
neural_net(3).b = zeros(1,r3);

%neural_net(4).W = W_scale * randn(r3, k);
%neural_net(4).b = zeros(1,k);
neural_net(4).W = W_scale * randn(r3, r4);
neural_net(4).b = zeros(1,r4);

neural_net(5).W = W_scale * randn(r4, k);
neural_net(5).b = zeros(1,k);

step_size = 1/20;                       % initial step size
lambda = 1/100;                        % regularization parameter
batchsize = 1000;
ERM = 0;
visualize = 0;

tic
neural_net = train_neural_net_SGD (Xtrain_normal, Ytrain, Xtest_normal, Ytest, neural_net, k, step_size, lambda, nb_iter, batchsize, ERM, visualize);
time_that_passed = toc;
[secs, minutes, hours, iterations] = elapsed_time(nb_iter, time_that_passed)

beep;