clear;
load('mnist_background_random_test.mat');
vname=@(x) inputname(1);

Xtrain = mnist_background_random_train(:,1:784);
Ytrain = mnist_background_random_train(:,785) + 1;
Xtest = mnist_background_random_test(:,1:784);
Ytest = mnist_background_random_test(:,785) + 1;

XNtrain = size(Xtrain,1)
YNtrain = size(Ytrain,1)
XNtest = size(Xtest,1)
YNtest = size(Ytest,1)

mu_Xtrain = mean(Xtrain,1);
stdev_Xtrain = std(Xtrain,0,1);

Xtrain_normal = (Xtrain - repmat(mu_Xtrain, [XNtrain, 1]) ) ./ repmat( stdev_Xtrain, [XNtrain,1]);
%Ytrain_normal = (Ytrain - repmat(mu_Xtrain, [YNtrain, 1]) ) ./ repmat( stdev_Xtrain, [YNtrain,1]);
Xtest_normal = (Xtest - repmat(mu_Xtrain, [XNtest, 1]) ) ./ repmat( stdev_Xtrain, [XNtest,1]);
%Ytest_normal = (Ytest - repmat(mu_Xtrain, [YNtest, 1]) ) ./ repmat( stdev_Xtrain, [YNtest,1]);

save('mnist_dirty_data', 'Xtrain', 'Ytrain', 'Xtest', 'Ytest', vname(mu_Xtrain), vname(stdev_Xtrain), 'Xtrain_normal', 'Xtest_normal' );