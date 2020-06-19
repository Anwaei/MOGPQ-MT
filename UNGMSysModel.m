classdef UNGMSysModel < AdditiveNoiseSystemModel
    %  System Model of Univariate Nonstationary Growth Model
    %  x(k) = 1/2x(k-1) + 25x(k-1)/(1+(x(k-1))^2) + 8cos(1.2k) + q(k-1); 
    properties
        k
        func
    end
    methods
        function obj = UNGMSysModel()
            %  Set state noise q(k-1) N(0,10)
            sysNoise = Gaussian(0,10);
            obj.setNoise(sysNoise);
            obj.k = 1;
            obj.func = @(x) 1/2*x + 25*x./(1+x.^2) + 8*cos(1.2*obj.k);
        end
        
        function predictedStates = systemEquation(obj, stateSamples)
            numSamples = size(stateSamples, 2);
            predictedStates = nan(1, numSamples);
            x = stateSamples(1,:);
            predictedStates(1,:) = obj.func(x);
        end
        
        function func = getFunc(obj)
            func = obj.func;
        end
    end
end

