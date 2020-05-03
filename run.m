% Toy
% y1 = x1cos(x2)
% y2 = x1sin(x2)
% SE kernel, alpha = 1, l = [60 6]

clear;

func_g = @(x) [x(1)*cos(x(2));x(1)*sin(x(2))];
% func_g = @(x) [x(1)*cos(x(2))];
func_g1 = @(x1,x2) x1.*cos(x2);
func_g2 = @(x1,x2) x1.*sin(x2);

conf.D = 2;
conf.Q = 2;

m = [4; 60/180*pi];
P = [0.5, 0; 0, 6/180*pi];
% func_integrand = @(x1,x2) multinormpdf2(x1, x2, m, P);
func_integrand1 = @(x1,x2) func_g1(x1,x2) .* multinormpdf2(x1, x2, m, P);
mu1_true = integral2(func_integrand1, -inf, inf, -inf, inf);
func_integrand2 = @(x1,x2) func_g2(x1,x2) .* multinormpdf2(x1, x2, m, P);
mu2_true = integral2(func_integrand2, -inf, inf, -inf, inf);
g = func_g(m);

% Sigma points
[Xi_s, N] = getSigmaPoints(conf.D, 'UKF');
conf.N = N;

% Kernel hypers
conf.cov = @covSEard;
alpha = 1; l = [2 1.5];
hyp = [log(l) log(alpha)];

[Mu, Pi, C] = GPQMT(m, P, hyp, Xi_s, func_g, conf);
