function [Mu, Pi, C] = GPQMT(m, P, hyp, Xi_s, g, obs_noise, conf)
% x: D x 1, p(x)=N(m,P)
% X_s: D x N sigma points
% Xi_s: D x N unit sigma points
% g: Q x 1, y = g(x), nonlinear function
% Mu: Q x 1, Mu = Ex[g(x)]
% Pi: Q x Q, Pi = Cx[g(x)]
% C: D x Q, C = Cx[x,g(x)]
% SE kernel

D = conf.D; N = conf.N; Q = conf.Q;
hyp0 = hyp;
Iq = eye(Q);

L = chol(P, 'lower');
X_s = m + L*Xi_s;

% evaluate function value
Y = zeros(Q,N);
for n = 1:N
    Y(:,n) = g(X_s(:,n)) + obs_noise.drawRndSamples(1);
end

% learn hyperparameters via ML
x_train = X_s;
y_train = Y(1,:);
ifmin = 0;
if ifmin == 1
    hyp = minimize(hyp0, @gp, -100, @infExact, conf.meanfunc, conf.covfunc,...
        conf.likfunc, x_train', y_train');
    fprintf('hyp0: alpha = %f, l = %f,%f, sigma2 = %f\n', exp(hyp0.cov(D+1)), exp(hyp0.cov(1:D)), exp(2*hyp0.lik));
    fprintf('hyp: alpha = %f, l = %f,%f, sigma2 = %f\n', exp(hyp.cov(D+1)), exp(hyp.cov(1:D)), exp(2*hyp.lik));
end

% evaluate kernel expectations
A = diag(exp(hyp.cov(1:D)));
A = inv(L'*inv(A)*L);  % !!!
I = eye(D);
Ainv = A\I;
alpha = exp(hyp.cov(D+1));
C1 = alpha^2 * det(Ainv+I)^(-1/2);
C2 = alpha^4 * det(2*Ainv+I)^(-1/2);

q = zeros(N,1); Q = zeros(N,N); R = zeros(D,N); % q: N x 1 Q: N x N R: D x N
kmean = alpha^2;
for n = 1:N
    xi_n = Xi_s(:,n);
    q(n) = C1*exp(-1/2*xi_n'/(A+I)*xi_n);
    R(:,n) = C1*exp(-1/2*xi_n'/(A+I)*xi_n)*((A+I)\xi_n);
    for m = 1:N
        xi_m = Xi_s(:,m);
        z = A\(xi_n+xi_m);
        Q(n,m) = C2*exp(-1/2*(xi_n'/A*xi_n + xi_m'/A*xi_m - z'/(2*Ainv+I)*z));
    end
end

Y = Y';  % In the paper of Jakub Pruher, Y is N x Q matrix.
K = conf.covfunc(hyp.cov, X_s');
sigma2 = exp(2*hyp.lik);
% K = conf.covfunc(hyp.cov, Xi_s');
Ik = eye(N);
K = K + Ik * sigma2;
Kinv = Ik/K;  % Solve the inverse of K (MAIN COMPUTATION COMPLEXITY)
w = Kinv*q;
W = Kinv*Q*Kinv;
Wc = R*Kinv;
s2 = kmean - trace(Q*Kinv);
Mu = Y'*w;
Pi = Y'*W*Y - Mu*Mu' + s2*Iq;
% Wa = diag(w);
% Pia = (Y-Mu)'*Wa*(Y-Mu);
C = L*Wc*Y;
end