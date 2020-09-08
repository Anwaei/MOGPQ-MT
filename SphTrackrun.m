clear;tic

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
confState.LMCsettings.weights = [1.0 0.0 0.0; 
                                 0.1 0.8 0.1; 
                                 0.2 0.2 0.6]';  % weights E x Q
disp('weights'); disp(confState.LMCsettings.weights);
confState.LMCsettings.gp = struct('covfunc',cell(E_state,1),'meanfunc',...
    cell(E_state,1),'hyp',cell(E_state,1));
[l_mostate,alpha] = setSEhypsState(E_state,confState.D,'mo'); disp('l_state_mo'); disp(l_mostate);
for e = 1:E_state  % set each gp
    confState.LMCsettings.gp(e).covfunc = @covSEard;
    confState.LMCsettings.gp(e).meanfunc = [];
    confState.LMCsettings.gp(e).hyp.cov = [log(l_mostate(e,:)) log(alpha(e,:))];
    confState.LMCsettings.gp(e).hyp.lik = log(sqrt(sigma2_noise));
end
confState.sample_cov_ref = sample_cov_ref;
confState.sample_method = 'UKF';

confMeas.model = 'LMC';
E_meas = 1; confMeas.LMCsettings.E = E_meas;  % num latent functions
confMeas.LMCsettings.weights = 1;
confMeas.LMCsettings.gp = struct('covfunc',cell(E_meas,1),'meanfunc',...
    cell(E_meas,1),'hyp',cell(E_meas,1));
[l_momeas,alpha] = setSEhypsMeas(E_meas,confMeas.D,'mo'); 
for e = 1:E_meas  % set each gp
    confMeas.LMCsettings.gp(e).covfunc = @covSEard;
    confMeas.LMCsettings.gp(e).meanfunc = [];
    confMeas.LMCsettings.gp(e).hyp.cov = [log(l_momeas(e,:)) log(alpha(e,:))];
    confMeas.LMCsettings.gp(e).hyp.lik = log(sqrt(sigma2_noise));
end
confMeas.sample_cov_ref = sample_cov_ref;
confMeas.sample_method = 'UKF';

confState_so = confState;
confState_so.LMCsettings.weights = [1 0 0; 0 1 0; 0 0 1];
[l_sostate,alpha] = setSEhypsState(E_state,confState_so.D,'so'); disp('l_state_so'); disp(l_sostate);
for e = 1:E_state  % set each gp
    confState_so.LMCsettings.gp(e).hyp.cov = [log(l_sostate(e,:)) log(alpha(e,:))];
    confState_so.LMCsettings.gp(e).hyp.lik = log(sqrt(sigma2_noise));
end

confMeas_so = confMeas;
confMeas_so.LMCsettings.weights = 1;
[l_someas,alpha] = setSEhypsMeas(E_meas,confMeas_so.D,'so'); disp('l_meas'); disp(l_someas);
for e = 1:E_meas  % set each gp
    confMeas_so.LMCsettings.gp(e).hyp.cov = [log(l_someas(e,:)) log(alpha(e,:))];
    confMeas_so.LMCsettings.gp(e).hyp.lik = log(sqrt(sigma2_noise));
end
%% Initialization
initialInference;

%% MC settings
numMC = 1;
RMSEState_ut_mc = zeros(1,15); RMSEState_mo_mc = zeros(1,15); RMSEState_so_mc = zeros(1,15);
JNEESState_ut_mc = zeros(1,15); JNEESState_mo_mc = zeros(1,15); JNEESState_so_mc = zeros(1,15);

%% Simulation
for m = 1:numMC
    initialInference;
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
        % ---- MO ----
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
        
        % ---- SO ----
        [data_train, confState_so] = generateTrainingData(updatedStateMean_GP_so,...
            updatedStateCov_GP_so, funcState, confState_so);  % [select sample methods]
        [predStateMean_GP_so, predStateCov_GP_so] = GPQMT_MO(updatedStateMean_GP_so,...
            updatedStateCov_GP_so, data_train, confState_so);
        predStateCov_GP_so = predStateCov_GP_so + stateNoiseCov;
        predStateMeansGP_so(:,k) = predStateMean_GP_so;
        predStateCovsGP_so(:,:,k) = predStateCov_GP_so;
        
        [data_train, confMeas_so] = generateTrainingData(predStateMean_GP_so,...
            predStateCov_GP_so, funcMeas, confMeas_so);  % [select sample methods]
        [predMeasMean_GP_so, predMeasCov_GP_so, predStateMeasCov_GP_so] = GPQMT_MO(predStateMean_GP_so,...
            predStateCov_GP_so, data_train, confMeas_so);
        predMeasCov_GP_so = predMeasCov_GP_so + measNoiseCov;
        predMeasMeansGP_so(:,k) = predMeasMean_GP_so;
        predMeasCovsGP_so(:,:,k) = predMeasCov_GP_so;
        predStateMeasCovsGP(:,:,k) = predStateMeasCov_GP_so;
        
        % Update
        KalmanGain_so = predStateMeasCov_GP_so / predMeasCov_GP_so;
        updatedStateMean_GP_so = predStateMean_GP_so + KalmanGain_so * (measurement - predMeasMean_GP_so);
        updatedStateCov_GP_so = predStateCov_GP_so - KalmanGain_so * predMeasCov_GP_so * KalmanGain_so';
        updatedStateMeansGP_so(:,k) = updatedStateMean_GP_so;
        updatedStateCovsGP_so(:,:,k) = updatedStateCov_GP_so;
        
        % disp(k);  
    end
    toc
    
    i = 1;  % UKF
    filter = filters.get(i);
    name   = filter.getName();
    
    updatedPosMean = reshape(updatedStateMeans(:,i,:), 3, numTimeSteps);
    RMSEState_ut = sqrt(1/numTimeSteps*sum(sum((updatedPosMean-sysStates).^2)));
    RMSEState_mo = sqrt(1/numTimeSteps*sum(sum((updatedStateMeansGP-sysStates).^2)));
    RMSEState_so = sqrt(1/numTimeSteps*sum(sum((updatedStateMeansGP_so-sysStates).^2)));
    fprintf('RMSEState_ut = %.4f, RMSEState_mo = %.4f, RMSEState_so = %.4f\n', ...
        RMSEState_ut, RMSEState_mo, RMSEState_so);
    
    NEESState_ut = zeros(1,numTimeSteps);
    NEESState_mo = zeros(1,numTimeSteps);
    NEESState_so = zeros(1,numTimeSteps);
    for kt = 1:numTimeSteps
        % UT
        err_ut = updatedPosMean(:,kt) - sysStates(:,kt);
        cov_ut = updatedStateCovs(:, :, i, kt);
        NEESState_ut(kt) = err_ut'/cov_ut*err_ut;
        % MO
        err_mo = updatedStateMeansGP(:,kt) - sysStates(:,kt);
        cov_mo = updatedStateCovsGP(:,:,kt);
        NEESState_mo(kt) = err_mo'/cov_mo*err_mo;
        % SO
        err_so = updatedStateMeansGP_so(:,kt) - sysStates(:,kt);
        cov_so = updatedStateCovsGP_so(:,:,kt);
        NEESState_so(kt) = err_so'/cov_so*err_so;
    end
    JNEESState_ut = sqrt(log(mean(NEESState_ut)/confState.Q)^2);
    JNEESState_mo = sqrt(log(mean(NEESState_mo)/confState.Q)^2);
    JNEESState_so = sqrt(log(mean(NEESState_so)/confState.Q)^2);
    fprintf('JNEESState_ut = %.4f, JNEESState_mo = %.4f, JNEESState_so = %.4f\n', ...
        JNEESState_ut, JNEESState_mo, JNEESState_so);
    
    RMSEState_ut_mc(m) = RMSEState_ut;
    RMSEState_mo_mc(m) = RMSEState_mo;
    RMSEState_so_mc(m) = RMSEState_so;
    JNEESState_ut_mc(m) = JNEESState_ut;
    JNEESState_mo_mc(m) = JNEESState_mo;
    JNEESState_so_mc(m) = JNEESState_so;
    
    fprintf('MC number: %d \n', m);
    
end

%% Results

% i = 1;  % UKF
% filter = filters.get(i);
% name   = filter.getName();
% 
% updatedPosMean = reshape(updatedStateMeans(:,i,:), 3, numTimeSteps);
% RMSEState_ut = sqrt(1/numTimeSteps*sum(sum((updatedPosMean-sysStates).^2)));
% RMSEState_mo = sqrt(1/numTimeSteps*sum(sum((updatedStateMeansGP-sysStates).^2)));
% RMSEState_so = sqrt(1/numTimeSteps*sum(sum((updatedStateMeansGP_so-sysStates).^2)));
% fprintf('RMSEState_ut = %.4f, RMSEState_mo = %.4f, RMSEState_so = %.4f\n', ...
%     RMSEState_ut, RMSEState_mo, RMSEState_so);
% 
% NEESState_ut = zeros(1,numTimeSteps);
% NEESState_mo = zeros(1,numTimeSteps);
% NEESState_so = zeros(1,numTimeSteps);
% for k = 1:numTimeSteps
%     % UT
%     err_ut = updatedPosMean(:,k) - sysStates(:,k);
%     cov_ut = updatedStateCovs(:, :, i, k);
%     NEESState_ut(k) = err_ut'/cov_ut*err_ut;
%     % MO
%     err_mo = updatedStateMeansGP(:,k) - sysStates(:,k);
%     cov_mo = updatedStateCovsGP(:,:,k);
%     NEESState_mo(k) = err_mo'/cov_mo*err_mo;
%     % SO
%     err_so = updatedStateMeansGP_so(:,k) - sysStates(:,k);
%     cov_so = updatedStateCovsGP_so(:,:,k);
%     NEESState_so(k) = err_so'/cov_so*err_so;    
% end
% JNEESState_ut = sqrt(log(mean(NEESState_ut)/confState.Q)^2);
% JNEESState_mo = sqrt(log(mean(NEESState_mo)/confState.Q)^2);
% JNEESState_so = sqrt(log(mean(NEESState_so)/confState.Q)^2);
% fprintf('JNEESState_ut = %.4f, JNEESState_mo = %.4f, JNEESState_so = %.4f\n', ...
%     JNEESState_ut, JNEESState_mo, JNEESState_so);

% stateLabel = {'position','velocity','ballistic parameter'};
% for figureNum = 1:confState.D
%     figure(figureNum)
%     stateNum = figureNum;
%     hold on
%     predPosMean = reshape(predStateMeans(stateNum,i,:), 1, numTimeSteps);
%     updatedPosMean = reshape(updatedStateMeans(stateNum,i,:), 1, numTimeSteps);
%     % updatedPosCovs = reshape(updatedStateCovs(:,:,i,:), confState.Q, confState.Q, numTimeSteps);
%     xlabel('time');
%     ylabel(stateLabel(stateNum));
%     title(['Estimate of ' name]);
%     plot(1:numTimeSteps, sysStates(stateNum,:), 'DisplayName','States');
% %    plot(1:numTimeSteps, measurements, 'DisplayName','Measurements');
%     plot(1:numTimeSteps, predPosMean, 'DisplayName','Predicted means');
%     plot(1:numTimeSteps, updatedPosMean, 'DisplayName','Updated means');
%     legend show;
% end

figure(20); 
stateLabel = {'position','velocity','ballistic parameter'};
StateLabel = {'Position','Velocity','Ballistic parameter'};
sposts = zeros(confState.D, numTimeSteps);
for n = 1:numTimeSteps
    sposts(:,n) = diag(updatedStateCovsGP(:,:,n));
end
for figureNum = confState.D+1:confState.D*2
    % figure(figureNum)
    stateNum = figureNum - confState.D;
    subplot(3,1,stateNum);
    if stateNum == 3
        axis([0 300 -2 4]);
    end
    hold on
    xlabel('time');
    ylabel(stateLabel(stateNum));
    titlename = strcat(StateLabel(stateNum),' state trajectory');
    title(titlename);
    ns = 1:numTimeSteps;
    mpost = updatedStateMeansGP(stateNum,:)';
    spost = (sqrt(sposts(stateNum,:)))';
    hi = patch([ns, fliplr(ns)],[mpost-2*spost; flipud(mpost+2*spost)], 1, 'FaceColor', [0.9,0.9,1], 'EdgeColor', 'none'); % This is the grey area in the plot.
    % patch([ns, fliplr(ns)],[mpost-sPost; flipud(mPost+sPost)], 1, 'FaceColor', [1,1,1]*0.8, 'EdgeColor', 'none'); % This is the grey area in the plot.
    set(hi,'handlevisibility','off');
    plot(1:numTimeSteps, updatedStateMeansGP(stateNum,:), 'DisplayName','Updated means');
    plot(1:numTimeSteps, sysStates(stateNum,:), 'DisplayName','True states');
    %    plot(1:numTimeSteps, measurements, 'DisplayName','Measurements');
    % plot(1:numTimeSteps, predStateMeansGP(stateNum,:), 'DisplayName','Predicted means');
    legend show;    
end


% measLabel = {'measure'};
% for figureNum = confState.D*2+1:confState.D*2+confMeas.Q
%     figure(figureNum)
%     measNum = figureNum - confState.D*2;
%     hold on
%     xlabel('time');
%     ylabel(measLabel(measNum));
%     title('Estimate of MOGP filter');
%     plot(1:numTimeSteps, measurements(measNum,:), 'DisplayName','Meas');
%     plot(1:numTimeSteps, predMeasMeansGP(measNum,:), 'DisplayName','Predicted measurements');
%     legend show;
% end
% stateLabel = {'position','velocity','ballistic parameter'};
% for figureNum = confState_so.D*2+confMeas_so.Q+1:confState_so.D*3+confMeas_so.Q
%     figure(figureNum)
%     stateNum = figureNum - (confState_so.D*2+confMeas_so.Q);
%     hold on
%     xlabel('time');
%     ylabel(stateLabel(stateNum));
%     title('Estimate of GPMT filter');
%     plot(1:numTimeSteps, sysStates(stateNum,:), 'DisplayName','States');
% %    plot(1:numTimeSteps, measurements, 'DisplayName','Measurements');
%     plot(1:numTimeSteps, predStateMeansGP_so(stateNum,:), 'DisplayName','Predicted means');
%     plot(1:numTimeSteps, updatedStateMeansGP_so(stateNum,:), 'DisplayName','Updated means');
%     legend show;
% end
% measLabel = {'measure'};
% for figureNum = confState_so.D*3+confMeas_so.Q+1:confState_so.D*3+confMeas_so.Q*2
%     figure(figureNum)
%     measNum = figureNum - (confState_so.D*3+confMeas_so.Q);
%     hold on
%     xlabel('time');
%     ylabel(measLabel(measNum));
%     title('Estimate of GPMT filter');
%     plot(1:numTimeSteps, measurements(measNum,:), 'DisplayName','Meas');
%     plot(1:numTimeSteps, predMeasMeansGP_so(measNum,:), 'DisplayName','Predicted measurements');
%     legend show;
% end


% figure;
% plot(1:numTimeSteps, NEESState_ut, 1:numTimeSteps, NEESState_mo, 1:numTimeSteps, NEESState_so);
% legend('ut','mo','so');
% 
% figure;
% plot(1:numTimeSteps, updatedPosMean(3,:), 1:numTimeSteps, updatedStateMeansGP(3,:),...
%     1:numTimeSteps, updatedStateMeansGP_so(3,:), 1:numTimeSteps, sysStates(3,:));
% legend('ut','mo','so','truth');

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