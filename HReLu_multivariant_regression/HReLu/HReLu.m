classdef HReLu
    %HSig class
    %   class that holds a 1 layered Hyper Sigmoid
    
    properties
        c % (K x D)
        t % (D+1 x K)
        lambda % (1 x 1)
    end
    
    methods
        function obj = HReLu(c,t,lambda)
            obj.c = c;
            obj.t = t;
            obj.lambda = lambda;
        end
        function any_nan_param = any_param_nan(obj)
            %returns true if any parameter is nan
            c_nan = any( isnan(obj.c) );
            t_nan = any( any( any( isnan(obj.t) ) ) );
            any_nan_param = c_nan | t_nan;
        end
        function [ F_X, Z, A ] = f(obj,X)
            Z = X * obj.t;  % (M * K) = (M x D+1) x (D+1 * K)
            A = max(0, Z); % (M * K)
            F_X = A * obj.c; % (M x D) = (M * K) * (K * D)
        end
        function [mdl] = gather(obj)
            mdl = HReLu(gather(obj.c), gather(obj.t), gather(obj.lambda));
        end
    end 
end