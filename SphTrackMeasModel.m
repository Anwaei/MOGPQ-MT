classdef SphTrackMeasModel < AdditiveNoiseMeasurementModel
    %  System Model of Univariate Nonstationary Growth Model
    %  z(k) = 1/20*x(k)^2 + r(k); 
    properties
        func
        sx
        sy
    end
    methods
        function obj = SphTrackMeasModel()
            %  Set state noise r(k) N(0,1)
            measNoise = Gaussian(0,9.2903e-4);
            obj.setNoise(measNoise);
            obj.sx = 30; obj.sy = 30;
            obj.func = @(states) sqrt(obj.sx^2 + (obj.sy-states(1,:)).^2);
        end
        
        function measurements = measurementEquation(obj, stateSamples)
%             numSamples = size(stateSamples, 2);
%             measurements = nan(1, numSamples);
%             p = stateSamples(1,:);
            measurements = obj.func(stateSamples);
        end
        
        function func = getFunc(obj)
            func = obj.func;
        end
    end
end

