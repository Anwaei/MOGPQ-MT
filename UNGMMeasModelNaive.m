classdef UNGMMeasModelNaive < AdditiveNoiseMeasurementModel
    %  System Model of Univariate Nonstationary Growth Model
    %  z(k) = 1/20*x(k)^2 + r(k); 
    methods
        function obj = UNGMMeasModelNaive()
            %  Set state noise r(k) N(0,1)
            measNoise = Gaussian(0,1);
            obj.setNoise(measNoise);
        end
        
        function measurements = measurementEquation(obj, stateSamples)
            numSamples = size(stateSamples, 2);
            measurements = nan(1, numSamples);
            x = stateSamples(1,:);
            % measurements(1,:) = 1/20*x.^2;
            measurements(1,:) = x;
        end
    end
end

