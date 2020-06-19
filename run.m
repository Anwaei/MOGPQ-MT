% Toy
% y1 = x1cos(x2)
% y2 = x1sin(x2)
% SE kernel, alpha = 1, l = [60 6]

clear;

% Function setting
a = 1;
func_g = @(x) [a*x(1,:).*cos(x(2,:));a*x(1,:).*sin(x(2,:))];
% func_g = @(x) [x(1)*cos(x(2))];
func_g1 = @(x1,x2) a*x1.*cos(x2);
func_g2 = @(x1,x2) a*x1.*sin(x2);
obsNoise = Gaussian([0;0],[0.5,0;0,0.5]);
% obsNoise = Gaussian(0,0.5);

conf.D = 2;  % num input
conf.Q = 2;  % num output

% Sigma points
[Xi_s, N] = getSigmaPoints(conf.D, '5th-Degree CKF');
conf.N = N;  % num sigma points

% GP settings
conf.covfunc = @covSEard;
conf.meanfunc = [];
conf.likfunc = @likGauss;
alpha = 1; l = [2 1.5];
hyp.cov = [log(l) log(alpha)];
hyp.lik = log(sqrt(0.4));

% MOGP settings
conf_mo = conf;
conf_mo.model = 'LMC';
E = 2; conf_mo.LMCsettings.E = E;  % num latent functions
conf_mo.LMCsettings.weights = rand(E, conf_mo.Q);  % weights of latent functions
conf_mo.LMCsettings.gp = struct('covfunc',cell(E,1),'meanfunc',cell(E,1),'hyp',cell(E,1));
[l,alpha] = setSEhyps(E,conf.D);
for e = 1:E  % set each gp
    conf_mo.LMCsettings.gp(e).covfunc = @covSEard;
    conf_mo.LMCsettings.gp(e).meanfunc = [];
    conf_mo.LMCsettings.gp(e).hyp.cov = [log(l(e,:)) log(alpha(e,:))];
    conf_mo.LMCsettings.gp(e).hyp.lik = log(sqrt(0.4));
end


numMC = 1000;
mtest = 1:1:5;
thetatest = 0/180*pi:20/180*pi:360/180*pi;
Nmtest = numel(mtest);
Nthetatest = numel(thetatest);
Ntest = Nmtest*Nthetatest;
mu_true = zeros(conf.Q,Ntest);
pi_true = zeros(conf.Q,conf.Q,Ntest);
mu_a = zeros(conf.Q,Ntest);
Pi_a = zeros(conf.Q,conf.Q,Ntest);
NEES = zeros(1,Ntest);
for i = 1:Nmtest
    for j = 1:Nthetatest
        k = (i-1)*Nthetatest + j;
        m = [mtest(i); thetatest(j)];
        P = [0.5, 0; 0, 6/180*pi];
        % m = [2;30/180*pi]; P = [0.5, 0; 0, 6/180*pi];
        % func_integrand = @(x1,x2) multinormpdf2(x1, x2, m, P);
%         func_integrand1 = @(x1,x2) func_g1(x1,x2) .* multinormpdf2(x1, x2, m, P);
%         mu1_true = integral2(func_integrand1, -inf, inf, -inf, inf);
%         func_integrand2 = @(x1,x2) func_g2(x1,x2) .* multinormpdf2(x1, x2, m, P);
%         mu2_true = integral2(func_integrand2, -inf, inf, -inf, inf);
%         mu_true(:,k) = [mu1_true;mu2_true];
%         g = func_g(m);
        
        xdistribution = Gaussian(m,P);
        x_mc = xdistribution.drawRndSamples(numMC);
        g_mc = func_g(x_mc) + obsNoise.drawRndSamples(numMC);
        mu_mc = mean(g_mc,2);
        pi_mc = cov(g_mc');
        mu_true(:,k) = mu_mc;
        pi_true(:,:,k) = pi_mc;
        
        [Mu, Pi, C] = GPQMT(m, P, hyp, Xi_s, func_g, obsNoise, conf);
        mu_a(:,k) = Mu;
        Pi_a(:,:,k) = Pi;
        
        errors = bsxfun(@minus, Mu, g_mc);
        NEESs = errors' / Pi * errors;
        % NEES(1,k) = (Mu - mu_true(:,k))' / Pi * (Mu - mu_true(:,k));
        NEES(1,k) = mean(diag(NEESs));
    end
end

k = 1:Ntest;
figure(1);
plot(k,mu_a(1,:),k,mu_true(1,:),k,mu_a(2,:),k,mu_true(2,:));
legend('mu1_evaluated','mu1_true','mu2_evaluated','mu2_true')
figure(2);
plot(k,NEES);
legend('NEES');
JNEES = sqrt(log(NEES./conf.D).^2);
figure(3);
plot(k,JNEES);
legend('JNEES');
RMSE = sqrt(1/Ntest*sum((mu_true(1,:)-mu_a(1,:)).^2 + ...
    (mu_true(2,:)-mu_a(2,:)).^2));
fprintf('RMSE: %f\n',RMSE);
MJNEES = mean(JNEES);
fprintf('MJNEES: %f\n',MJNEES);
