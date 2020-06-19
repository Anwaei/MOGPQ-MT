classdef SphTrackSysModel < AdditiveNoiseSystemModel
    %  System Model of spherical object falling tracking
    %  p(k+1) = p(k) - dt*v(k) + q1(k);
    %  v(k+1) = v(k) - dt*v(k)^2*theta(k)*exp(-gamma*p(k)) + q2(k);
    %  theta(k+1) = theta(k) + q3(k);
    methods
        function obj = SphTrackSysModel()
            %  Set state noise q(k-1) N(0,10)
            sysNoise = Gaussian(zeros(3,1),[]);
            obj.setNoise(sysNoise);
        end
        
        function predictedStates = systemEquation(obj, stateSamples)
            numSamples = size(stateSamples, 2);
            predictedStates = nan(1, numSamples);
            x = stateSamples(1,:);
            predictedStates(1,:) = 1/2*x + 25*x./(1+x.^2);
        end
    end
end

