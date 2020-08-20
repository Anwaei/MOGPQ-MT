clear;

%% Environment setting
confState.D = 3; confState.Q = 3;
confMeas.D = 3; confMeas.Q = 1;
sigma2_noise = 1e-8;
obs_noise = Gaussian(zeros(confState.Q,1),sigma2_noise*eye(confState.Q));
confState.obs_noise = obs_noise;
obs_noise = Gaussian(zeros(confMeas.Q,1),sigma2_noise*eye(confMeas.Q));
confMeas.obs_noise = obs_noise;

sysModel = SphTrackSysModel();
measModel = SphTrackMeasModel();

%% MOGP setting
sample_cov_ref = diag([1,1,0.8]);

confState.model = 'LMC';
E_state = 3; confState.LMCsettings.E = E_state;  % num latent functions
confState.LMCsettings.weights = [1 0 0; 0 1 0; 0 0 1];  % weights E x Q
confState.LMCsettings.gp = struct('covfunc',cell(E_state,1),'meanfunc',...
    cell(E_state,1),'hyp',cell(E_state,1));
[l,alpha] = setSEhypsState(E_state,confState.D);
for e = 1:E_state  % set each gp
    confState.LMCsettings.gp(e).covfunc = @covSEard;
    confState.LMCsettings.gp(e).meanfunc = [];
    confState.LMCsettings.gp(e).hyp.cov = [log(l(e,:)) log(alpha(e,:))];
    confState.LMCsettings.gp(e).hyp.lik = log(sqrt(sigma2_noise));
end
confState.sample_cov_ref = sample_cov_ref;

confMeas.model = 'LMC';
E_meas = 1; confMeas.LMCsettings.E = E_meas;  % num latent functions
confMeas.LMCsettings.weights = 1;
confMeas.LMCsettings.gp = struct('covfunc',cell(E_meas,1),'meanfunc',...
    cell(E_meas,1),'hyp',cell(E_meas,1));
[l,alpha] = setSEhypsMeas(E_meas,confMeas.D);
for e = 1:E_meas  % set each gp
    confMeas.LMCsettings.gp(e).covfunc = @covSEard;
    confMeas.LMCsettings.gp(e).meanfunc = [];
    confMeas.LMCsettings.gp(e).hyp.cov = [log(l(e,:)) log(alpha(e,:))];
    confMeas.LMCsettings.gp(e).hyp.lik = log(sqrt(sigma2_noise));
end
confMeas.sample_cov_ref = sample_cov_ref;

%% Filters setting
filters = FilterSet();

filter = EKF();
% filter.setColor({ 'Color', [0 0.5 0] });
filters.add(filter);

filter = UKF();
% filter.setColor({ 'Color', 'r' });
filters.add(filter);

filter = UKF('Iterative UKF');
filter.setMaxNumIterations(5);
% filter.setColor({ 'Color', 'b' });
filters.add(filter);

filter = SIRPF();
filter.setNumParticles(10^5);
filters.add(filter);

filter = CKF();
filters.add(filter);

numFilters = filters.getNumFilters();

numTimeSteps = 300;

initialState = Gaussian([90;6;1.7],diag([0.0929 1.4865 1]));
initialStateTrue = Gaussian([90;6;1.5],diag([0.0929 1.4865 1e-4]));
[initialMean, initialCov] = initialState.getMeanAndCov();
sysStates = nan(confState.D,numTimeSteps);
measurements = nan(confMeas.Q,numTimeSteps);

% Filters
updatedStateMeans  = nan(confState.D, numFilters, numTimeSteps);
updatedStateCovs   = nan(confState.D, confState.D, numFilters, numTimeSteps);
predStateMeans     = nan(confState.D, numFilters, numTimeSteps);
predStateCovs      = nan(confState.D, confState.D, numFilters, numTimeSteps);
runtimesUpdate     = nan(numFilters, numTimeSteps);
runtimesPrediction = nan(numFilters, numTimeSteps);

%% GP Filters
predStateMeansGP = nan(confState.D, numTimeSteps);
predStateCovsGP = nan(confState.D, confState.D, numTimeSteps);
predMeasMeansGP = nan(confMeas.Q, numTimeSteps);
predMeasCovsGP = nan(confMeas.Q, confMeas.Q, numTimeSteps);
predStateMeasCovsGP = nan(confMeas.D, confMeas.Q, numTimeSteps);
updatedStateMeansGP = nan(confState.D, numTimeSteps);
updatedStateCovsGP = nan(confState.D, confState.D, numTimeSteps);

updatedStateMean_GP = initialMean;
updatedStateCov_GP = initialCov;

%% Simulation
filters.setStates(initialState);
sysState = initialStateTrue.drawRndSamples(1);
sysState = initialStateTrue.getMeanAndCov();
% sysState = 0.1;
disp('Simulation start');

for k = 1:numTimeSteps
    
    % sysModel.k = k;
    
    % Perform state prediction
    runtimesPrediction(:, k) = filters.predict(sysModel);    
    [predStateMeans(:, :, k), ...
        predStateCovs(:, :, :, k)] = filters.getStatesMeanAndCov();
    
    % Simulate next system state
    sysState = sysModel.simulate(sysState);
    
    % Simulate measurement for time step k
    measurement = measModel.simulate(sysState);
    
    % Save data
    sysStates(:, k)    = sysState;
    measurements(:, k) = measurement;
    
    % Perform measurement update
    runtimesUpdate(:, k) = filters.update(measModel, measurement);    
    [updatedStateMeans(:, :, k), ...
        updatedStateCovs(:, :, :, k)] = filters.getStatesMeanAndCov();
    
    
%    ------ GPQ implementation ------
    funcState = sysModel.getFunc();
    [~,stateNoiseCov] = sysModel.noise.getMeanAndCov();
    funcMeas = measModel.getFunc();
    [~,measNoiseCov] = measModel.noise.getMeanAndCov();
    
    % Prediction
    [data_train, confState] = generateTrainingData(updatedStateMean_GP,...
        updatedStateCov_GP, funcState, confState);  % [select sample methods]
    [predStateMean_GP, predStateCov_GP] = GPQMT_MO(updatedStateMean_GP,...
        updatedStateCov_GP, data_train, confState);
    predStateCov_GP = predStateCov_GP + stateNoiseCov;
    predStateMeansGP(:,k) = predStateMean_GP;
    predStateCovsGP(:,:,k) = predStateCov_GP;
    
    [data_train, confMeas] = generateTrainingData(predStateMean_GP,... 
        predStateCov_GP, funcMeas, confMeas);  % [select sample methods]
    [predMeasMean_GP, predMeasCov_GP, predStateMeasCov_GP] = GPQMT_MO(predStateMean_GP,...
        predStateCov_GP, data_train, confMeas);
    predMeasCov_GP = predMeasCov_GP + measNoiseCov;
    predMeasMeansGP(:,k) = predMeasMean_GP;
    predMeasCovsGP(:,:,k) = predMeasCov_GP;
    predStateMeasCovsGP(:,:,k) = predStateMeasCov_GP;
    
    % Update
    KalmanGain = predStateMeasCov_GP / predMeasCov_GP;
    updatedStateMean_GP = predStateMean_GP + KalmanGain * (measurement - predMeasMean_GP);
    updatedStateCov_GP = predStateCov_GP - KalmanGain * predMeasCov_GP * KalmanGain';
    updatedStateMeansGP(:,k) = updatedStateMean_GP;
    updatedStateCovsGP(:,:,k) = updatedStateCov_GP;
    
    disp(k);
end

%% Results

i = 5;  % UKF
filter = filters.get(i);
name   = filter.getName();

stateLabel = {'position','velocity','ballistic parameter'};
for figureNum = 1:confState.D
    figure(figureNum)
    stateNum = figureNum;
    hold on
    predPosMean = reshape(predStateMeans(stateNum,i,:), 1, numTimeSteps);
    updatedPosMean = reshape(updatedStateMeans(stateNum,i,:), 1, numTimeSteps);
    % updatedPosCovs = reshape(updatedStateCovs(:,:,i,:), confState.Q, confState.Q, numTimeSteps);
    xlabel('time');
    ylabel(stateLabel(stateNum));
    title(['Estimate of ' name]);
    plot(1:numTimeSteps, sysStates(stateNum,:), 'DisplayName','States');
%    plot(1:numTimeSteps, measurements, 'DisplayName','Measurements');
    plot(1:numTimeSteps, predPosMean, 'DisplayName','Predicted means');
    plot(1:numTimeSteps, updatedPosMean, 'DisplayName','Updated means');
    legend show;
end

stateLabel = {'position','velocity','ballistic parameter'};
for figureNum = confState.D+1:confState.D*2
    figure(figureNum)
    stateNum = figureNum - confState.D;
    hold on
    xlabel('time');
    ylabel(stateLabel(stateNum));
    title('Estimate of MOGP filter');
    plot(1:numTimeSteps, sysStates(stateNum,:), 'DisplayName','States');
%    plot(1:numTimeSteps, measurements, 'DisplayName','Measurements');
    plot(1:numTimeSteps, predStateMeansGP(stateNum,:), 'DisplayName','Predicted means');
    plot(1:numTimeSteps, updatedStateMeansGP(stateNum,:), 'DisplayName','Updated means');
    legend show;
end
measLabel = {'measure'};
for figureNum = confState.D*2+1:confState.D*2+confMeas.Q
    figure(figureNum)
    measNum = figureNum - confState.D*2;
    hold on
    xlabel('time');
    ylabel(measLabel(measNum));
    title('Estimate of MOGP filter');
    plot(1:numTimeSteps, measurements(measNum,:), 'DisplayName','Meas');
    plot(1:numTimeSteps, predMeasMeansGP(measNum,:), 'DisplayName','Predicted measurements');
    legend show;
end


% figureNum = figureNum + 1;    
% figure(figureNum);
% 
% for i = 1:numFilters
%     
%     filter = filters.get(i);
%     name   = filter.getName();
%     
%     predPosMean = reshape(predStateMeans(:,i,:), confState.Q, numTimeSteps);
%     updatedPosMean = reshape(updatedStateMeans(:,i,:), confState.Q, numTimeSteps);
%     updatedPosCovs = reshape(updatedStateCovs(:,:,i,:), confState.Q, confState.Q, numTimeSteps);
%     
%     % Plot states
%     subplot(2,3,i);
%     hold on;
%     xlabel('time');
%     ylabel('x');    
%     title(['Estimate of ' name]);
%     
%     plot(1:numTimeSteps, sysStates, 'DisplayName','States');
%     plot(1:numTimeSteps, measurements, 'DisplayName','Measurements');
%     plot(1:numTimeSteps, predPosMean, 'DisplayName','Predicted means');
%     plot(1:numTimeSteps, updatedPosMean, 'DisplayName','Updated means');
%     legend show;
%     
%     % Compute RMSE and NEES
%     errorStates = updatedPosMean-sysStates;
%     RMSE = sqrt(1/numTimeSteps*sum(sum(errorStates.^2)));
%     fprintf('RMSE of %s: %f\n', name, RMSE);
%     NEES = nan(1,numTimeSteps);
%     for k = 1:numTimeSteps
%         NEES(k) = errorStates(:,k)'*updatedPosCovs(:,:,k)*errorStates(:,k);
%     end
%     JNEES = sqrt((log(sum(NEES)./confState.D)).^2)/numTimeSteps;
%     fprintf('JNEES of %s: %f\n', name, JNEES);
% end
% 
% figure;
% hold on
% plot(1:numTimeSteps, sysStates, 'DisplayName','States');
% plot(1:numTimeSteps, measurements, 'DisplayName','Measurements');
% plot(1:numTimeSteps, predStateMeansGP, 'DisplayName','Predicted means');
% plot(1:numTimeSteps, updatedStateMeansGP, 'DisplayName','Updated means');
% legend show;
% xlabel('time');
% ylabel('x');