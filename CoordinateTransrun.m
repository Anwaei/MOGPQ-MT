% Toy
% y1 = x1cos(x2)
% y2 = x1sin(x2)
% SE kernel, alpha = 1, l = [60 6]

clear; close all;
tic

%% Function setting
a = 1;
func_g = @(x) [a*x(1,:).*cos(x(2,:));a*x(1,:).*sin(x(2,:))];
% func_g = @(x) [x(1)*cos(x(2))];
func_g1 = @(x1,x2) a*x1.*cos(x2);
func_g2 = @(x1,x2) a*x1.*sin(x2);

conf_mo.D = 2;  % num input
conf_mo.Q = 2;  % num output
obs_noise = Gaussian([0;0],[1e-10,0;0,1e-10]);
conf_mo.obs_noise = obs_noise;  % observation noise
%% MOGP settings
conf_mo.model = 'LMC';
E = 2; conf_mo.LMCsettings.E = E;  % num latent functions
% conf_mo.LMCsettings.weights = rand(E, conf_mo.Q);  % weights of latent functions
conf_mo.LMCsettings.weights = [1, -0.4; -0.4, 1];
% conf_mo.LMCsettings.weights = [1,1];
conf_mo.LMCsettings.gp = struct('covfunc',cell(E,1),'meanfunc',cell(E,1),'hyp',cell(E,1));
[l,alpha] = setSEhyps(E,conf_mo.D, 'mo');
for e = 1:E  % set each gp
    conf_mo.LMCsettings.gp(e).covfunc = @covSEard;
    conf_mo.LMCsettings.gp(e).meanfunc = [];
    conf_mo.LMCsettings.gp(e).hyp.cov = [log(l(e,:)) log(alpha(e,:))];
    conf_mo.LMCsettings.gp(e).hyp.lik = log(sqrt(0.4));
end
conf_mo.sample_cov_ref = eye(conf_mo.Q);
conf_mo.sample_method = 'UKF';

conf_so = conf_mo;
conf_so.LMCsettings.weights = [1, 0; 0, 1];
[l,alpha] = setSEhyps(E,conf_so.D, 'so');
for e = 1:E  % set each gp
    conf_so.LMCsettings.gp(e).hyp.cov = [log(l(e,:)) log(alpha(e,:))];
    conf_so.LMCsettings.gp(e).hyp.lik = log(sqrt(0.4));
end

num_MC = 1500;

% %% Run1: fix covariance, test on means
% mx1_test = 1:1:5;
% mx2_test = 0/180*pi:60/180*pi:360/180*pi;
% vx1_fix = 0.1:0.2:1.1;
% vx2_fix = 6/180*pi:6/180*pi:30/180*pi;
% 
% N_mx1t = numel(mx1_test);
% N_mx2t = numel(mx2_test);
% N_vx1f = numel(vx1_fix);
% N_vx2f = numel(vx2_fix);
% 
% N_mtest = N_mx1t*N_mx2t;
% N_vfix = N_vx1f*N_vx2f;
% 
% RMSE_mo_vfix = zeros(1,N_vfix);
% JNEES_mo_vfix = zeros(1,N_vfix);
% RMSE_so_vfix = zeros(1,N_vfix);
% JNEES_so_vfix = zeros(1,N_vfix);
% RMSE_ut_vfix = zeros(1,N_vfix);
% JNEES_ut_vfix = zeros(1,N_vfix);
% 
% for nv1 = 1:N_vx1f
%     for nv2 = 1:N_vx2f        
%         nv = (nv1-1)*N_vx2f + nv2;        
%         mu_true = zeros(conf_mo.Q,N_mtest);
%         Pi_true = zeros(conf_mo.Q,conf_mo.Q,N_mtest);
%         mu_a_mo = zeros(conf_mo.Q,N_mtest);
%         Pi_a_mo = zeros(conf_mo.Q,conf_mo.Q,N_mtest);
%         NEES_mo = zeros(1,N_mtest);
%         mu_a_so = zeros(conf_so.Q,N_mtest);
%         Pi_a_so = zeros(conf_so.Q,conf_so.Q,N_mtest);
%         NEES_so = zeros(1,N_mtest);
%         mu_a_ut = zeros(conf_so.Q,N_mtest);
%         Pi_a_ut = zeros(conf_so.Q,conf_so.Q,N_mtest);
%         NEES_ut = zeros(1,N_mtest);
%         for i = 1:N_mx1t
%             for j = 1:N_mx2t
%                 k = (i-1)*N_mx2t + j;
%                 m = [mx1_test(i); mx2_test(j)];
%                 P = [vx1_fix(nv1), 0; 0, vx2_fix(nv2)];
%                 
%                 xdistribution = Gaussian(m,P);
%                 x_mc = xdistribution.drawRndSamples(num_MC);
%                 g_mc = func_g(x_mc) + obs_noise.drawRndSamples(num_MC);
%                 mu_mc = mean(g_mc,2);
%                 pi_mc = cov(g_mc');
%                 mu_true(:,k) = mu_mc;
%                 Pi_true(:,:,k) = pi_mc;
%                 
%                 [data_train_mo, conf_mo] = generateTrainingData(m, P, func_g, conf_mo);  % MOGPQ
%                 [mu_mo, Pi_mo, C_mo] = GPQMT_MO(m, P, data_train_mo, conf_mo);
%                 mu_a_mo(:,k) = mu_mo;
%                 Pi_a_mo(:,:,k) = Pi_mo;
%                 
%                 errors_mo = bsxfun(@minus, mu_mo, g_mc);
%                 NEESs_mo = errors_mo' / Pi_mo * errors_mo;
%                 % NEES(1,k) = (Mu - mu_true(:,k))' / Pi * (Mu - mu_true(:,k));
%                 NEES_mo(1,k) = mean(diag(NEESs_mo));
%                 
%                 [data_train_so, conf_so] = generateTrainingData(m, P, func_g, conf_so);  % GPQ
%                 [mu_so, Pi_so, C_so] = GPQMT_MO(m, P, data_train_so, conf_so);
%                 mu_a_so(:,k) = mu_so;
%                 Pi_a_so(:,:,k) = Pi_so;
%                 
%                 errors_so = bsxfun(@minus, mu_so, g_mc);
%                 NEESs_so = errors_so' / Pi_so * errors_so;
%                 NEES_so(1,k) = mean(diag(NEESs_so));
%                 
%                 [mu_ut, Pi_ut] = CooUT(m, P, func_g, conf_mo);  % UT
%                 mu_a_ut(:,k) = mu_ut;
%                 Pi_a_ut(:,:,k) = Pi_ut;
%                 errors_ut = bsxfun(@minus, mu_ut, g_mc);
%                 NEESs_ut = errors_ut' / Pi_ut * errors_ut;
%                 NEES_ut(1,k) = mean(diag(NEESs_ut));
%             end
%         end
%         RMSE_mo = sqrt(1/N_mtest*sum(sum((mu_true-mu_a_mo).^2)));  % Performance of MOGPQ
%         RMSE_mo_vfix(nv) = RMSE_mo;
%         JNEES_mo = sqrt(log(mean(NEES_mo)/conf_mo.Q)^2);
%         JNEES_mo_vfix(nv) = JNEES_mo;
%         RMSE_so = sqrt(1/N_mtest*sum(sum((mu_true-mu_a_so).^2)));  % Performance of GPQ
%         RMSE_so_vfix(nv) = RMSE_so;
%         JNEES_so = sqrt(log(mean(NEES_so)/conf_so.Q)^2);
%         JNEES_so_vfix(nv) = JNEES_so;
%         RMSE_ut = sqrt(1/N_mtest*sum(sum((mu_true-mu_a_ut).^2)));  % Performance of UT
%         RMSE_ut_vfix(nv) = RMSE_ut;
%         JNEES_ut = sqrt(log(mean(NEES_ut)/conf_mo.Q)^2);
%         JNEES_ut_vfix(nv) = JNEES_ut;
%         fprintf('n_vfix=%d\n', nv);
%     end
% end
% 
% %% Run2: fix means, test on covariances
% mx1_fix = 1:1:5;
% mx2_fix = 0/180*pi:60/180*pi:360/180*pi;
% vx1_test = 0.1:0.2:1.1;
% vx2_test = 6/180*pi:6/180*pi:30/180*pi;
% 
% N_mx1f = numel(mx1_fix);
% N_mx2f = numel(mx2_fix);
% N_vx1t = numel(vx1_test);
% N_vx2t = numel(vx2_test);
% 
% N_mfix = N_mx1f*N_mx2f;
% N_vtest = N_vx1t*N_vx2t;
% 
% RMSE_mo_mfix = zeros(1,N_mfix);
% JNEES_mo_mfix = zeros(1,N_mfix);
% RMSE_so_mfix = zeros(1,N_mfix);
% JNEES_so_mfix = zeros(1,N_mfix);
% RMSE_ut_mfix = zeros(1,N_mfix);
% JNEES_ut_mfix = zeros(1,N_mfix);
% 
% for nm1 = 1:N_mx1f
%     for nm2 = 1:N_mx2f        
%         nm = (nm1-1)*N_mx2f + nm2;        
%         mu_true = zeros(conf_mo.Q,N_mtest);
%         Pi_true = zeros(conf_mo.Q,conf_mo.Q,N_mtest);
%         mu_a_mo = zeros(conf_mo.Q,N_mtest);
%         Pi_a_mo = zeros(conf_mo.Q,conf_mo.Q,N_mtest);
%         NEES_mo = zeros(1,N_mtest);
%         mu_a_so = zeros(conf_so.Q,N_mtest);
%         Pi_a_so = zeros(conf_so.Q,conf_so.Q,N_mtest);
%         NEES_so = zeros(1,N_mtest);
%         mu_a_ut = zeros(conf_so.Q,N_mtest);
%         Pi_a_ut = zeros(conf_so.Q,conf_so.Q,N_mtest);
%         NEES_ut = zeros(1,N_mtest);
%         for i = 1:N_vx1t
%             for j = 1:N_vx2t
%                 k = (i-1)*N_vx2t + j;
%                 m = [mx1_fix(nm1); mx2_fix(nm2)];
%                 P = [vx1_test(i), 0; 0, vx2_test(j)];
%                 
%                 xdistribution = Gaussian(m,P);
%                 x_mc = xdistribution.drawRndSamples(num_MC);
%                 g_mc = func_g(x_mc) + obs_noise.drawRndSamples(num_MC);
%                 mu_mc = mean(g_mc,2);
%                 pi_mc = cov(g_mc');
%                 mu_true(:,k) = mu_mc;
%                 Pi_true(:,:,k) = pi_mc;
%                 
%                 [data_train_mo, conf_mo] = generateTrainingData(m, P, func_g, conf_mo);  % MOGPQ
%                 [mu_mo, Pi_mo, C_mo] = GPQMT_MO(m, P, data_train_mo, conf_mo);
%                 mu_a_mo(:,k) = mu_mo;
%                 Pi_a_mo(:,:,k) = Pi_mo;
%                 
%                 errors_mo = bsxfun(@minus, mu_mo, g_mc);
%                 NEESs_mo = errors_mo' / Pi_mo * errors_mo;
%                 % NEES(1,k) = (Mu - mu_true(:,k))' / Pi * (Mu - mu_true(:,k));
%                 NEES_mo(1,k) = mean(diag(NEESs_mo));
%                 
%                 [data_train_so, conf_so] = generateTrainingData(m, P, func_g, conf_so);  % GPQ
%                 [mu_so, Pi_so, C_so] = GPQMT_MO(m, P, data_train_so, conf_so);
%                 mu_a_so(:,k) = mu_so;
%                 Pi_a_so(:,:,k) = Pi_so;
%                 
%                 errors_so = bsxfun(@minus, mu_so, g_mc);
%                 NEESs_so = errors_so' / Pi_so * errors_so;
%                 NEES_so(1,k) = mean(diag(NEESs_so));
%                 
%                 [mu_ut, Pi_ut] = CooUT(m, P, func_g, conf_mo);  % UT
%                 mu_a_ut(:,k) = mu_ut;
%                 Pi_a_ut(:,:,k) = Pi_ut;
%                 errors_ut = bsxfun(@minus, mu_ut, g_mc);
%                 NEESs_ut = errors_ut' / Pi_ut * errors_ut;
%                 NEES_ut(1,k) = mean(diag(NEESs_ut));
%             end
%         end
%         RMSE_mo = sqrt(1/N_mtest*sum(sum((mu_true-mu_a_mo).^2)));  % Performance of MOGPQ
%         RMSE_mo_mfix(nm) = RMSE_mo;
%         JNEES_mo = sqrt(log(mean(NEES_mo)/conf_mo.Q)^2);
%         JNEES_mo_mfix(nm) = JNEES_mo;
%         RMSE_so = sqrt(1/N_mtest*sum(sum((mu_true-mu_a_so).^2)));  % Performance of GPQ
%         RMSE_so_mfix(nm) = RMSE_so;
%         JNEES_so = sqrt(log(mean(NEES_so)/conf_so.Q)^2);
%         JNEES_so_mfix(nm) = JNEES_so;
%         RMSE_ut = sqrt(1/N_mtest*sum(sum((mu_true-mu_a_ut).^2)));  % Performance of UT
%         RMSE_ut_mfix(nm) = RMSE_ut;
%         JNEES_ut = sqrt(log(mean(NEES_ut)/conf_mo.Q)^2);
%         JNEES_ut_mfix(nm) = JNEES_ut;
%         fprintf('n_mfix=%d\n', nm);
%     end
% end
% for weight_trial = -5:5
%     conf_mo.LMCsettings.weights = [1, weight_trial/10; weight_trial/10, 1];
%% Run3: missing data test, vfix
mx1_test = 1:1:5;
mx2_test = 0/180*pi:60/180*pi:360/180*pi;
vx1_fix = 0.1:0.2:1.1;
vx2_fix = 6/180*pi:6/180*pi:30/180*pi;

N_mx1t = numel(mx1_test);
N_mx2t = numel(mx2_test);
N_vx1f = numel(vx1_fix);
N_vx2f = numel(vx2_fix);

N_mtest = N_mx1t*N_mx2t;
N_missing_vf = N_vx1f*N_vx2f;

RMSE_mo_missing_vf = zeros(1,N_missing_vf);
JNEES_mo_missing_vf = zeros(1,N_missing_vf);
RMSE_so_missing_vf = zeros(1,N_missing_vf);
JNEES_so_missing_vf = zeros(1,N_missing_vf);

for nv1 = 1:N_vx1f
    for nv2 = 1:N_vx2f        
        n_missing_vf = (nv1-1)*N_vx2f + nv2;        
        mu_true = zeros(conf_mo.Q,N_mtest);
        Pi_true = zeros(conf_mo.Q,conf_mo.Q,N_mtest);
        mu_a_mo = zeros(conf_mo.Q,N_mtest);
        Pi_a_mo = zeros(conf_mo.Q,conf_mo.Q,N_mtest);
        NEES_mo = zeros(1,N_mtest);
        mu_a_so = zeros(conf_so.Q,N_mtest);
        Pi_a_so = zeros(conf_so.Q,conf_so.Q,N_mtest);
        NEES_so = zeros(1,N_mtest);
        for i = 1:N_mx1t
            for j = 1:N_mx2t
                k = (i-1)*N_mx2t + j;
                m = [mx1_test(i); mx2_test(j)];
                P = [vx1_fix(nv1), 0; 0, vx2_fix(nv2)];
                
                xdistribution = Gaussian(m,P);
                x_mc = xdistribution.drawRndSamples(num_MC);
                g_mc = func_g(x_mc) + obs_noise.drawRndSamples(num_MC);
                mu_mc = mean(g_mc,2);
                pi_mc = cov(g_mc');
                mu_true(:,k) = mu_mc;
                Pi_true(:,:,k) = pi_mc;
                
                [data_train_mo, conf_mo] = generateMissingTrainingData(m, P, func_g, conf_mo);  % MOGPQ
                [mu_mo, Pi_mo, C_mo] = GPQMT_MO(m, P, data_train_mo, conf_mo);
                mu_a_mo(:,k) = mu_mo;
                Pi_a_mo(:,:,k) = Pi_mo;
                
                errors_mo = bsxfun(@minus, mu_mo, g_mc);
                NEESs_mo = errors_mo' / Pi_mo * errors_mo;
                % NEES(1,k) = (Mu - mu_true(:,k))' / Pi * (Mu - mu_true(:,k));
                NEES_mo(1,k) = mean(diag(NEESs_mo));
                
                [data_train_so, conf_so] = generateMissingTrainingData(m, P, func_g, conf_so);  % GPQ
                [mu_so, Pi_so, C_so] = GPQMT_MO(m, P, data_train_so, conf_so);
                mu_a_so(:,k) = mu_so;
                Pi_a_so(:,:,k) = Pi_so;
                
                errors_so = bsxfun(@minus, mu_so, g_mc);
                NEESs_so = errors_so' / Pi_so * errors_so;
                NEES_so(1,k) = mean(diag(NEESs_so));
            end
        end
        RMSE_mo = sqrt(1/N_mtest*sum(sum((mu_true-mu_a_mo).^2)));  % Performance of MOGPQ
        RMSE_mo_missing_vf(n_missing_vf) = RMSE_mo;
        JNEES_mo = sqrt(log(mean(NEES_mo)/conf_mo.Q)^2);
        JNEES_mo_missing_vf(n_missing_vf) = JNEES_mo;
        RMSE_so = sqrt(1/N_mtest*sum(sum((mu_true-mu_a_so).^2)));  % Performance of GPQ
        RMSE_so_missing_vf(n_missing_vf) = RMSE_so;
        JNEES_so = sqrt(log(mean(NEES_so)/conf_so.Q)^2);
        JNEES_so_missing_vf(n_missing_vf) = JNEES_so;
        fprintf('n_missing_vf=%d, RMSE_mo=%.4f, RMSE_so=%.4f, JNEES_mo=%.4f, JNEES_mo=%.4f\n'...
            , n_missing_vf, RMSE_mo, RMSE_so, JNEES_mo, JNEES_so);
    end
end

%% Run4: missing data test, mfix
mx1_fix = 1:1:5;
mx2_fix = 0/180*pi:60/180*pi:360/180*pi;
vx1_test = 0.1:0.2:1.1;
vx2_test = 6/180*pi:6/180*pi:30/180*pi;

N_mx1f = numel(mx1_fix);
N_mx2f = numel(mx2_fix);
N_vx1t = numel(vx1_test);
N_vx2t = numel(vx2_test);

N_vtest = N_vx1t*N_vx2t;
N_missing_mf = N_mx1f*N_mx2f;

RMSE_mo_missing_mf = zeros(1,N_missing_mf);
JNEES_mo_missing_mf = zeros(1,N_missing_mf);
RMSE_so_missing_mf = zeros(1,N_missing_mf);
JNEES_so_missing_mf = zeros(1,N_missing_mf);

for nm1 = 1:N_mx1f
    for nm2 = 1:N_mx2f           
        n_missing_mf = (nm1-1)*N_mx2f + nm2;        
        mu_true = zeros(conf_mo.Q,N_mtest);
        Pi_true = zeros(conf_mo.Q,conf_mo.Q,N_mtest);
        mu_a_mo = zeros(conf_mo.Q,N_mtest);
        Pi_a_mo = zeros(conf_mo.Q,conf_mo.Q,N_mtest);
        NEES_mo = zeros(1,N_mtest);
        mu_a_so = zeros(conf_so.Q,N_mtest);
        Pi_a_so = zeros(conf_so.Q,conf_so.Q,N_mtest);
        NEES_so = zeros(1,N_mtest);
        for i = 1:N_vx1t
            for j = 1:N_vx2t
                k = (i-1)*N_vx2t + j;
                m = [mx1_fix(nm1); mx2_fix(nm2)];
                P = [vx1_test(i), 0; 0, vx2_test(j)];
                
                xdistribution = Gaussian(m,P);
                x_mc = xdistribution.drawRndSamples(num_MC);
                g_mc = func_g(x_mc) + obs_noise.drawRndSamples(num_MC);
                mu_mc = mean(g_mc,2);
                pi_mc = cov(g_mc');
                mu_true(:,k) = mu_mc;
                Pi_true(:,:,k) = pi_mc;
                
                [data_train_mo, conf_mo] = generateMissingTrainingData(m, P, func_g, conf_mo);  % MOGPQ
                [mu_mo, Pi_mo, C_mo] = GPQMT_MO(m, P, data_train_mo, conf_mo);
                mu_a_mo(:,k) = mu_mo;
                Pi_a_mo(:,:,k) = Pi_mo;
                
                errors_mo = bsxfun(@minus, mu_mo, g_mc);
                NEESs_mo = errors_mo' / Pi_mo * errors_mo;
                % NEES(1,k) = (Mu - mu_true(:,k))' / Pi * (Mu - mu_true(:,k));
                NEES_mo(1,k) = mean(diag(NEESs_mo));
                     
                [data_train_so, conf_so] = generateMissingTrainingData(m, P, func_g, conf_so);  % GPQ
                [mu_so, Pi_so, C_so] = GPQMT_MO(m, P, data_train_so, conf_so);
                mu_a_so(:,k) = mu_so;
                Pi_a_so(:,:,k) = Pi_so;
                
                errors_so = bsxfun(@minus, mu_so, g_mc);
                NEESs_so = errors_so' / Pi_so * errors_so;
                NEES_so(1,k) = mean(diag(NEESs_so));
                
            end
        end
        RMSE_mo = sqrt(1/N_mtest*sum(sum((mu_true-mu_a_mo).^2)));  % Performance of MOGPQ
        RMSE_mo_missing_mf(n_missing_mf) = RMSE_mo;
        JNEES_mo = sqrt(log(mean(NEES_mo)/conf_mo.Q)^2);
        JNEES_mo_missing_mf(n_missing_mf) = JNEES_mo;
        RMSE_so = sqrt(1/N_mtest*sum(sum((mu_true-mu_a_so).^2)));  % Performance of GPQ
        RMSE_so_missing_mf(n_missing_mf) = RMSE_so;
        JNEES_so = sqrt(log(mean(NEES_so)/conf_so.Q)^2);
        JNEES_so_missing_mf(n_missing_mf) = JNEES_so;
        fprintf('n_missing_mf=%d, RMSE_mo=%.4f, RMSE_so=%.4f, JNEES_mo=%.4f, JNEES_mo=%.4f\n'...
            , n_missing_mf, RMSE_mo, RMSE_so, JNEES_mo, JNEES_so);
    end
end

%% Run5: test on different E
mx1_testE = 1:1:5;
mx2_testE = 0/180*pi:60/180*pi:360/180*pi;
vx1_testE = 0.1:0.2:1.1;
vx2_testE = 6/180*pi:6/180*pi:36/180*pi;
N_mx1 = numel(mx1_testE);
N_mx2 = numel(mx2_testE);
N_vx1 = numel(vx1_testE);
N_vx2 = numel(vx2_testE);
N_testE = N_mx1*N_mx2*N_vx1*N_vx2;
maxE = 6;
RMSE_mo_testEs = zeros(1,maxE);
JNEES_mo_testEs = zeros(1,maxE);
conf_mo_testEs = generateLMCsets(maxE);
for e = 1:maxE
    conf_mo = conf_mo_testEs{e};
    mu_true = zeros(conf_mo.Q,N_testE);
    Pi_true = zeros(conf_mo.Q,conf_mo.Q,N_testE);
    mu_a_mo = zeros(conf_mo.Q,N_testE);
    Pi_a_mo = zeros(conf_mo.Q,conf_mo.Q,N_testE);
    NEES_mo = zeros(1,N_testE);
    k = 1;
    for nm1 = 1:N_mx1
        for nm2 = 1:N_mx2
            for nv1 = 1:N_vx1
                for nv2 = 1:N_vx2
                    m = [mx1_testE(nm1); mx2_testE(nm2)];
                    P = [vx1_testE(nv1), 0; 0, vx2_testE(nv2)];
                    
                    xdistribution = Gaussian(m,P);
                    x_mc = xdistribution.drawRndSamples(num_MC);
                    g_mc = func_g(x_mc) + obs_noise.drawRndSamples(num_MC);
                    mu_mc = mean(g_mc,2);
                    pi_mc = cov(g_mc');
                    mu_true(:,k) = mu_mc;
                    Pi_true(:,:,k) = pi_mc;
                    
                    [data_train_mo, conf_mo] = generateMissingTrainingData(m, P, func_g, conf_mo);  % MOGPQ
                    [mu_mo, Pi_mo, C_mo] = GPQMT_MO(m, P, data_train_mo, conf_mo);
                    mu_a_mo(:,k) = mu_mo;
                    Pi_a_mo(:,:,k) = Pi_mo;
                    
                    errors_mo = bsxfun(@minus, mu_mo, g_mc);
                    NEESs_mo = errors_mo' / Pi_mo * errors_mo;
                    NEES_mo(1,k) = mean(diag(NEESs_mo));
                    
                    %disp(k);
                    k = k+1;
                end
            end
        end
    end
    disp(e);
    RMSE_mo_testE = sqrt(1/N_testE*sum(sum((mu_true-mu_a_mo).^2)));
    JNEES_mo_testE = sqrt(log(mean(NEES_mo)/conf_mo.Q)^2);
    RMSE_mo_testEs(e) = RMSE_mo_testE;
    JNEES_mo_testEs(e) = JNEES_mo_testE;
end


%% Figures
% figure(1); title('results with UT sigma points input'); set(gcf,'unit','centimeters','position',[5 5 16 14])
% subplot(2,2,1); hold on; title('average RMSE over means'); xlabel({'$$l$$'},'Interpreter','latex'); ylabel('RMSE');
% k = 1:N_vfix; plot(k,RMSE_mo_vfix,k,RMSE_so_vfix,k,RMSE_ut_vfix); ylim([0.02 0.07]);
% legend('MOGPQ', 'GPQ', 'UT', 'location', 'NorthWest');
% subplot(2,2,2); hold on; title('average JNEES over means'); xlabel({'$$l$$'},'Interpreter','latex'); ylabel('JNEES');
% k = 1:N_vfix; plot(k,JNEES_mo_vfix,k,JNEES_so_vfix,k,JNEES_ut_vfix);
% legend('MOGPQ', 'GPQ', 'UT', 'location', 'NorthWest');
% subplot(2,2,3); hold on; title('average RMSE over covariances'); xlabel({'$$n$$'},'Interpreter','latex'); ylabel('RMSE');
% k = 1:N_mfix; plot(k,RMSE_mo_mfix,k,RMSE_so_mfix,k,RMSE_ut_mfix); xlim([0 35]);
% legend('MOGPQ', 'GPQ', 'UT', 'location', 'NorthWest');
% subplot(2,2,4); hold on; title('average JNEES over covariances'); xlabel({'$$n$$'},'Interpreter','latex'); ylabel('JNEES');
% k = 1:N_mfix; plot(k,JNEES_mo_mfix,k,JNEES_so_mfix,k,JNEES_ut_mfix); xlim([0 35]); ylim([0 0.4]);
% legend('MOGPQ', 'GPQ', 'UT', 'location', 'NorthWest');
figure(2); title('results with data missing input'); set(gcf,'unit','centimeters','position',[6 6 16 14])
subplot(2,2,1); hold on; title('average RMSE over means'); xlabel('$\mathrm{Index\ of\ } \mathbf{P}$','Interpreter','latex'); ylabel('RMSE');
k = 1:N_missing_vf; plot(k,RMSE_mo_missing_vf,k,RMSE_so_missing_vf); ylim([0 1]);
legend('MOGPQ', 'GPQ', 'location', 'NorthWest');
subplot(2,2,2); hold on; title('average JNEES over means'); xlabel('$\mathrm{Index\ of\ } \mathbf{P}$','Interpreter','latex'); ylabel('JNEES');
k = 1:N_missing_vf; plot(k,JNEES_mo_missing_vf,k,JNEES_so_missing_vf); ylim([0 0.5]);
legend('MOGPQ', 'GPQ', 'location', 'NorthWest');
subplot(2,2,3); hold on; title('average RMSE over covariances'); xlabel('$\mathrm{Index\ of\ } \mathbf{m}$','Interpreter','latex'); ylabel('RMSE');
k = 1:N_missing_mf; plot(k,RMSE_mo_missing_mf,k,RMSE_so_missing_mf); xlim([0 35]);
legend('MOGPQ', 'GPQ', 'location', 'NorthWest');
subplot(2,2,4); hold on; title('average JNEES over covariances');xlabel('$\mathrm{Index\ of\ } \mathbf{m}$','Interpreter','latex'); ylabel('JNEES');
k = 1:N_missing_mf; plot(k,JNEES_mo_missing_mf,k,JNEES_so_missing_mf); xlim([0 35]); ylim([0 1.3]);
legend('MOGPQ', 'GPQ', 'location', 'NorthWest');

figure(3); % title('Performance with different numbers of latent functions')
subplot(1,2,1); bar(2:maxE, RMSE_mo_testEs(2:end)); title('$\mathrm{RMSE\ with\ different\ } E$','Interpreter','latex');
xlabel('$E$','Interpreter','latex'); ylabel('RMSE');
subplot(1,2,2); bar(2:maxE, JNEES_mo_testEs(2:end)); title('$\mathrm{JNEES\ with\ different\ } E$','Interpreter','latex');
xlabel('$E$','Interpreter','latex'); ylabel('JNEES');

toc
% end