classdef cross_validation_data < handle
    %
    
    properties
        X
        y
        per_train
        per_cv
        X_train
        X_cv
        X_test
        y_train
        y_cv
        y_test
    end
    
    methods
        function obj = cross_validation_data(X,y,per_train,per_cv)
            obj.X = X;
            obj.y = y;
            obj.per_train = per_train;
            obj.per_cv = per_cv;
            %% split_data_for_hold_out_cross_validation
            [~, N] = size(obj.X);
            N_train = floor(N * obj.per_train);
            N_cv = floor(N * obj.per_cv);
            obj.X_train = obj.X(:,1:N_train);
            obj.X_cv = obj.X(:, N_train+1:N_train+N_cv );
            obj.X_test = obj.X(:, N_train+N_cv+1:N);
            obj.y_train = obj.y(:,1:N_train);
            obj.y_cv = obj.y(:, N_train+1:N_train+N_cv );
            obj.y_test = obj.y(:, N_train+N_cv+1:N);
        end
        function [ X_train,X_cv,X_test, y_train,y_cv,y_test ] = get_data_for_hold_out_cross_validation(obj)
            X_train = obj.X_train;
            X_cv = obj.X_cv;
            X_test = obj.X_test;
            y_train = obj.y_train;
            y_cv = obj.y_cv;
            y_test = obj.y_test;
        end
        function [] = change_data_sets(obj, X_train,X_cv,X_test, y_train,y_cv,y_test)
            obj.X_train = X_train;
            obj.X_cv = X_cv;
            obj.X_test = X_test;
            obj.y_train = y_train;
            obj.y_cv = y_cv;
            obj.y_test = y_test;
        end
        % Make a copy of a handle object.
        function new = copy(this)
            % Instantiate new object of the same class.
            new = feval(class(this));
 
            % Copy all non-hidden properties.
            p = properties(this);
            for i = 1:length(p)
                new.(p{i}) = this.(p{i});
            end
        end
        function [] = normalize_data(obj)
            obj.X = normc(obj.X);
            obj.y = normc(obj.y);
            obj.X_train = normc(obj.X_train);
            obj.X_cv = normc(obj.X_cv);
            obj.X_test = normc(obj.X_test);
            obj.y_train = normc(obj.y_train);
            obj.y_cv = normc(obj.y_cv);
            obj.y_test = normc(obj.y_test);
        end
    end
    methods (Static)
        function [X_new, y_new] = shuffle_data(X,y)
            [~, N] = size(X);
            permute_ordering = randperm(N);
            X_new = X(:, permute_ordering);
            y_new = y(:,permute_ordering);
        end
    end
   
end

