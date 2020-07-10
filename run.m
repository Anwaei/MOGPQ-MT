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

conf_mo.D = 2;  % num input
conf_mo.Q = 2;  % num output
obs_noise = Gaussian([0;0],[1e-10,0;0,1e-10]);
conf_mo.obs_noise = obs_noise;  % observation noise
% MOGP settings
conf_mo.model = 'LMC';
E = 2; conf_mo.LMCsettings.E = E;  % num latent functions
% conf_mo.LMCsettings.weights = rand(E, conf_mo.Q);  % weights of latent functions
conf_mo.LMCsettings.weights = [0.8, 0.2; 0.2, 0.8];
% conf_mo.LMCsettings.weights = [1,1];
conf_mo.LMCsettings.gp = struct('covfunc',cell(E,1),'meanfunc',cell(E,1),'hyp',cell(E,1));
[l,alpha] = setSEhyps(E,conf_mo.D);
for e = 1:E  % set each gp
    conf_mo.LMCsettings.gp(e).covfunc = @covSEard;
    conf_mo.LMCsettings.gp(e).meanfunc = [];
    conf_mo.LMCsettings.gp(e).hyp.cov = [log(l(e,:)) log(alpha(e,:))];
    conf_mo.LMCsettings.gp(e).hyp.lik = log(sqrt(0.4));
end

m = [2; 30/180*pi];
P = [0.5, 0; 0, 6/180*pi];
% MOGPQMT learning
conf_mo.LMCsettings.weights = [-2, -1; -2, 0.5];
[data_train, conf_mo] = generateTrainingData(m, P, func_g, conf_mo);
[mu, Pi, C] = GPQMT_MO(m, P, data_train, conf_mo);

% Definiteness test
[data_train, conf_mo] = generateTrainingData(m, P, func_g, conf_mo);
k = 1;
for a = 2:-0.5:-2
    for b = 2:-0.5:-2
        for c = 2:-0.5:-2
            for d = 2:-0.5:-2
                if (a~=0||b~=0)&&(c~=0||d~=0)&&abs(det([a,b;c,d]))>1e-15
                    conf_mo.LMCsettings.weights = [a, b; c, d];
                    [mu, Pi, C] = GPQMT_MO(m, P, data_train, conf_mo);
                    eig_Pi = eig(Pi);
                    min_eig(k) = min(eig_Pi);
                    if min_eig(k)<0
                        fprintf('k=%d\n',k);
                        fprintf('a=%3.2f,b=%3.2f,c=%3.2f,d=%3.2f\n',a,b,c,d);
                        fprintf('min_eig=%f\n',min_eig(k));
                        disp(Pi);
                    end
                    k = k+1;
                end
            end
        end
    end
end

numMC = 1000;
mtest = 1:1:5;
thetatest = 0/180*pi:20/180*pi:360/180*pi;
Nmtest = numel(mtest);
Nthetatest = numel(thetatest);
Ntest = Nmtest*Nthetatest;
mu_true = zeros(conf_mo.Q,Ntest);
pi_true = zeros(conf_mo.Q,conf_mo.Q,Ntest);
mu_a = zeros(conf_mo.Q,Ntest);
Pi_a = zeros(conf_mo.Q,conf_mo.Q,Ntest);
NEES = zeros(1,Ntest);
for i = 1:Nmtest
    for j = 1:Nthetatest
        k = (i-1)*Nthetatest + j;
        m = [mtest(i); thetatest(j)];
        P = [0.5, 0; 0, 6/180*pi];
        
        xdistribution = Gaussian(m,P);
        x_mc = xdistribution.drawRndSamples(numMC);
        g_mc = func_g(x_mc) + obs_noise.drawRndSamples(numMC);
        mu_mc = mean(g_mc,2);
        pi_mc = cov(g_mc');
        mu_true(:,k) = mu_mc;
        pi_true(:,:,k) = pi_mc;
        
        [data_train, conf_mo] = generateTrainingData(m, P, func_g, conf_mo);
        [Mu, Pi, C] = GPQMT_MO(m, P, data_train, conf_mo);
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
JNEES = sqrt(log(NEES./conf_mo.D).^2);
figure(3);
plot(k,JNEES);
legend('JNEES');
RMSE = sqrt(1/Ntest*sum((mu_true(1,:)-mu_a(1,:)).^2 + ...
    (mu_true(2,:)-mu_a(2,:)).^2));
fprintf('RMSE: %f\n',RMSE);
MJNEES = mean(JNEES);
fprintf('MJNEES: %f\n',MJNEES);
