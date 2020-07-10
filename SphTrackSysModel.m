classdef SphTrackSysModel < AdditiveNoiseSystemModel
    %  System Model of spherical object falling tracking
    %  p(k+1) = p(k) - dt*v(k) + q1(k);
    %  v(k+1) = v(k) - dt*v(k)^2*theta(k)*exp(-gamma*p(k)) + q2(k);
    %  theta(k+1) = theta(k) + q3(k);
    properties
        dt
        gamma
        func
    end
    methods
        function obj = SphTrackSysModel()
            sysNoise = Gaussian(zeros(3,1),1e-6*eye(3));  % Set state noise q(k)
            obj.setNoise(sysNoise);
            obj.dt = 0.1;  % Set dt
            obj.gamma = 0.164;
            obj.func = @(states) [states(1,:)-obj.dt*states(2,:); ...
                states(2,:)-obj.dt*states(2,:).^2.*states(3,:).*exp(-obj.gamma*states(1,:));...
                states(3,:)];
        end
        
        function predictedStates = systemEquation(obj, stateSamples)
%             numSamples = size(stateSamples, 2);
%             predictedStates = nan(3, numSamples);
%             p = stateSamples(1,:);
%             v = stateSamples(2,:);
%             theta = stateSamples(3,:);
%             predictedStates = obj.func(p,v,theta);
            predictedStates = obj.func(stateSamples);
        end
        
        function func = getFunc(obj)
            func = obj.func;
        end
    end
end

