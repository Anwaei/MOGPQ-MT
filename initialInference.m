%% Filters setting
filters = FilterSet();

filter = EKF();
% filter.setColor({ 'Color', [0 0.5 0] });
filters.add(filter);

filter = UKF();
% filter.setColor({ 'Color', 'r' });
filters.add(filter);

% filter = UKF('Iterative UKF');
% filter.setMaxNumIterations(5);
% % filter.setColor({ 'Color', 'b' });
% filters.add(filter);

% filter = SIRPF();
% filter.setNumParticles(10^5);
% filters.add(filter);

% filter = CKF();
% filters.add(filter);

numFilters = filters.getNumFilters();

numTimeSteps = 300;

initialState = Gaussian([90;6;1.7],diag([0.0929 1.4865 10]));
initialStateTrue = Gaussian([90;6;1.5],diag([0.0929 1.4865 1e-4]));
initialState = Gaussian([91.44;6.1;1.7],diag([0.092 1.486 10]));
initialStateTrue = Gaussian([91.44;6.1;1.5],diag([0.0929 1.486 1e-4]));
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

%% GP setting
% MO
predStateMeansGP = nan(confState.D, numTimeSteps);
predStateCovsGP = nan(confState.D, confState.D, numTimeSteps);
predMeasMeansGP = nan(confMeas.Q, numTimeSteps);
predMeasCovsGP = nan(confMeas.Q, confMeas.Q, numTimeSteps);
predStateMeasCovsGP = nan(confMeas.D, confMeas.Q, numTimeSteps);
updatedStateMeansGP = nan(confState.D, numTimeSteps);
updatedStateCovsGP = nan(confState.D, confState.D, numTimeSteps);

updatedStateMean_GP = initialMean;
updatedStateCov_GP = initialCov;

% SO
predStateMeansGP_so = nan(confState_so.D, numTimeSteps);
predStateCovsGP_so = nan(confState_so.D, confState_so.D, numTimeSteps);
predMeasMeansGP_so = nan(confMeas_so.Q, numTimeSteps);
predMeasCovsGP_so = nan(confMeas_so.Q, confMeas_so.Q, numTimeSteps);
predStateMeasCovsGP_so = nan(confMeas_so.D, confMeas_so.Q, numTimeSteps);
updatedStateMeansGP_so = nan(confState_so.D, numTimeSteps);
updatedStateCovsGP_so = nan(confState_so.D, confState_so.D, numTimeSteps);

updatedStateMean_GP_so = initialMean;
updatedStateCov_GP_so = initialCov;