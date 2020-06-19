classdef UNGMMeasModel < AdditiveNoiseMeasurementModel
    %  System Model of Univariate Nonstationary Growth Model
    %  z(k) = 1/20*x(k)^2 + r(k); 
    properties
        func
    end
    methods
        function obj = UNGMMeasModel()
            %  Set state noise r(k) N(0,1)
            measNoise = Gaussian(0,1);
            obj.setNoise(measNoise);
            obj.func = @(x) 1/20*x.^2;
        end
        
        function measurements = measurementEquation(obj, stateSamples)
            numSamples = size(stateSamples, 2);
            measurements = nan(1, numSamples);
            x = stateSamples(1,:);
            measurements(1,:) = obj.func(x);
            % measurements(1,:) = sqrt(x.^2);
        end
        
        function func = getFunc(obj)
            func = obj.func;
        end
    end
end

