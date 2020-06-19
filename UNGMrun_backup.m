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

filters.setStates(initialState);
% sysState = initialState.drawRndSamples(1);
sysState = 0.2;

for k = 1:numTimeSteps
    
    sysModel.k = k;
    
    % Simulate measurement for time step k
    measurement = measModel.simulate(sysState);
    
    % Save data
    sysStates(:, k)    = sysState;
    measurements(:, k) = measurement;
    
    % Perform measurement update
    runtimesUpdate(:, k) = filters.update(measModel, measurement);    
    [updatedStateMeans(:, :, k), ...
        updatedStateCovs(:, :, :, k)] = filters.getStatesMeanAndCov();
    
    % Simulate next system state
    sysState = sysModel.simulate(sysState);
    
    % Perform state prediction
    runtimesPrediction(:, k) = filters.predict(sysModel);    
    [predStateMeans(:, :, k), ...
        predStateCovs(:, :, :, k)] = filters.getStatesMeanAndCov();
    
    
    % GPQ implementation
    funcState = sysModel.getFunc();
    funcMeas = measModel.getFunc();
    
    % Update
    
    % Prediction
    updatedStateMean_GP = updatedStateMeansGP(:,k);
    updatedStateCov_GP = updatedStateCovsGP(:,k);
    [predStateMean_GP, predStateCov_GP] = GPQMT(updatedStateMean_GP,...
        updatedStateCov_GP, hypState, Xi_s, funcState, conf);
    predStateCov_GP = predStateCov_GP + stateNoiseCov;
    predStateMeansGP(:,k) = predStateMean_GP;
    predStateCovsGP(:,k) = predStateCov_GP;
    
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
    JNEES = sqrt((log(sum(NEES)./conf.Q)).^2);
    fprintf('JNEES of %s: %f\n', name, JNEES);
end