clear;

conf.D = 1;
conf.Q = 1;

sysModel = UNGMSysModel();
measModel = UNGMMeasModel();

% Filters setup
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

numTimeSteps = 100;

initialState = Gaussian(0,2);
[initialMean, initialCov] = initialState.getMeanAndCov();
sysStates = nan(conf.D,numTimeSteps);
measurements = nan(conf.Q,numTimeSteps);

% Filters
updatedStateMeans  = nan(conf.D, numFilters, numTimeSteps);
updatedStateCovs   = nan(conf.D, conf.D, numFilters, numTimeSteps);
predStateMeans     = nan(conf.D, numFilters, numTimeSteps);
predStateCovs      = nan(conf.D, conf.D, numFilters, numTimeSteps);
runtimesUpdate     = nan(numFilters, numTimeSteps);
runtimesPrediction = nan(numFilters, numTimeSteps);

% GP Filters
predStateMeansGP = nan(conf.D, numTimeSteps);
predStateCovsGP = nan(conf.D, conf.D, numTimeSteps);
predMeasMeansGP = nan(conf.Q, numTimeSteps);
predMeasCovsGP = nan(conf.Q, conf.Q, numTimeSteps);
predStateMeasCovsGP = nan(conf.D, conf.Q, numTimeSteps);
updatedStateMeansGP = nan(conf.D, numTimeSteps);
updatedStateCovsGP = nan(conf.D, conf.D, numTimeSteps);

updatedStateMean_GP = initialMean;
updatedStateCov_GP = initialCov;

[Xi_s, N] = getSigmaPoints(conf.D, '5th-Degree CKF');
conf.N = N;
conf.covfunc = @covSEard;
alpha = 1; l = 2.1;
hypState.cov = [log(l) log(alpha)];
hypState.lik = log(sqrt(1e-6));
hypMeas.cov = [log(l) log(alpha)];
hypMeas.lik = log(sqrt(1e-6));

filters.setStates(initialState);
% sysState = initialState.drawRndSamples(1);
sysState = 0.1;

for k = 1:numTimeSteps
    
    sysModel.k = k;
    
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
    obs_noise = Gaussian(0,1e-6);
    
    % Prediction
    [predStateMean_GP, predStateCov_GP] = GPQMT(updatedStateMean_GP,...
        updatedStateCov_GP, hypState, Xi_s, funcState, obs_noise, conf);
    predStateCov_GP = predStateCov_GP + stateNoiseCov;
    predStateMeansGP(:,k) = predStateMean_GP;
    predStateCovsGP(:,:,k) = predStateCov_GP;
    
    [predMeasMean_GP, predMeasCov_GP, predStateMeasCov_GP] = GPQMT(predStateMean_GP,...
        predStateCov_GP, hypMeas, Xi_s, funcMeas, obs_noise, conf);
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
    
end

figure;

for i = 1:numFilters
    
    filter = filters.get(i);
    name   = filter.getName();
    
    predPosMean = reshape(predStateMeans(:,i,:), conf.Q, numTimeSteps);
    updatedPosMean = reshape(updatedStateMeans(:,i,:), conf.Q, numTimeSteps);
    updatedPosCovs = reshape(updatedStateCovs(:,:,i,:), conf.Q, conf.Q, numTimeSteps);
    
    % Plot states
    subplot(2,3,i);
    hold on;
    xlabel('time');
    ylabel('x');    
    title(['Estimate of ' name]);
    
    plot(1:numTimeSteps, sysStates, 'DisplayName','States');
    plot(1:numTimeSteps, measurements, 'DisplayName','Measurements');
    plot(1:numTimeSteps, predPosMean, 'DisplayName','Predicted means');
    plot(1:numTimeSteps, updatedPosMean, 'DisplayName','Updated means');
    legend show;
    
    % Compute RMSE and NEES
    errorStates = updatedPosMean-sysStates;
    RMSE = sqrt(1/numTimeSteps*sum(errorStates.^2));
    fprintf('RMSE of %s: %f\n', name, RMSE);
    NEES = nan(1,numTimeSteps);
    for k = 1:numTimeSteps
        NEES(k) = errorStates(:,k)'*updatedPosCovs(:,:,k)*errorStates(:,k);
    end
    JNEES = sqrt((log(sum(NEES)./conf.Q)).^2)/numTimeSteps;
    fprintf('JNEES of %s: %f\n', name, JNEES);
end

figure;
hold on
plot(1:numTimeSteps, sysStates, 'DisplayName','States');
plot(1:numTimeSteps, measurements, 'DisplayName','Measurements');
plot(1:numTimeSteps, predStateMeansGP, 'DisplayName','Predicted means');
plot(1:numTimeSteps, updatedStateMeansGP, 'DisplayName','Updated means');
legend show;
xlabel('time');
ylabel('x');