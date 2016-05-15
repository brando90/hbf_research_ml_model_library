clear;clc;
%% data
D = 3
N = 5
X = rand(D, N);
X = magic(N); %% <------ uncomment this line for disaster
%% process data
x_mean = mean(X, 2); %% computes the mean of the data x_mean = sum(x^(i))
X_centered = X - repmat(x_mean, [1,N]);
%% PCA
[coeff, score, latent, ~, ~, mu] = pca(X'); % coeff =  U
[U, S, V] = svd(X_centered); % coeff = U
%%
U
coeff
V
%% Reconstruct data
% if U = coeff then the following should be an identity I (since U is orthonormal)
UU_T =  U * U'
coeff_coeff_T = coeff * coeff'
% if U = coeff then they should be able to perfectly reconstruct the data
X_tilde_U = U * U'*X
X_tilde_coeff = coeff*coeff'*X

%X_tilde_score = ( score * coeff' + repmat(mu,[N,1]) )'

% latent
% latent.^(2)
% latent.^(0.5)
% S
% S.^(2)
% S.^(0.5)