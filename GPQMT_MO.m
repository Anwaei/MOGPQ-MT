function [mu, Pi, C] = GPQMT_MO(m, P, data_train, conf_mo)
% x: D x 1, p(x)=N(m,P)
% X_s: D x N sigma points
% Xi_s: D x N unit sigma points
% g: Q x 1, y = g(x), nonlinear function
% Mu: Q x 1, Mu = Ex[g(x)]
% Pi: Q x Q, Pi = Cx[g(x)]
% C: D x Q, C = Cx[x,g(x)]
% SE kernel

Q = conf_mo.Q; D = conf_mo.D;
L = chol(P, 'lower');

xi_sigma = data_train.xi_sigma;
x_sigma = data_train.x_sigma;
y = data_train.y_train;
[~, NQ] = size(xi_sigma); N = NQ/Q;

[E1, E2, E3, E4] = solveExpectations(m,P,xi_sigma,conf_mo);

Kx = createGramMatrix(x_sigma, conf_mo);
[~,sigma_o] = conf_mo.obs_noise.getMeanAndCov(); In = eye(N);
Kx = Kx + kron(In,sigma_o);
% Kx = Kx + kron(sigma_o,In);
I = eye(size(Kx));Kx_inv = I/Kx;
Kv = Kx_inv*(y*y')*Kx_inv - Kx_inv;
Kv1 = Kx_inv*(y*y')*Kx_inv; Kv2 = Kx_inv;
Kc = y'*Kx_inv;

E_Kx = E2;
E_Kxx = E1;
E_Kv = zeros(Q,Q); E_Kc = zeros(D,Q);
for p = 1:Q
    for q = 1:Q
        E_Kv(p,q) = trace(E3(:,:,p,q)*Kv);
    end
    E_Kc(:,p) = E4(:,:,p) * Kc';
end

mu = E_Kx * Kx_inv * y;
Pi = E_Kv + E_Kxx - mu*mu';
% Pi = (Pi+Pi')/2;
C = L * E_Kc;
end

% ----------- test -----------
% E_Kv1 = zeros(Q,Q);
% for p = 1:Q
%     for q = 1:Q
%         E_Kv1(p,q) = trace(E3(:,:,p,q)*Kv1);
%     end
% end
% E_Kv2 = zeros(Q,Q);
% for p = 1:Q
%     for q = 1:Q
%         E_Kv2(p,q) = trace(E3(:,:,p,q)*Kv2);
%     end
% end
% Pi_1 = E_Kv1 - mu*mu';
% Pi_2 = - E_Kv2 + E_Kxx;

% % ----------- test -----------
% E = conf_mo.LMCsettings.E; 
% GPs =  conf_mo.LMCsettings.gp;
% weights = conf_mo.LMCsettings.weights;
% covfunc = conf_mo.LMCsettings.gp(1).covfunc;
% N_test = 10000;
% sampling = GaussianSamplingRnd();
% sampling.setNumSamples(N_test);
% xi_test = sampling.getStdNormalSamples(D);
% L = chol(P, 'lower');
% x_test = m + L*xi_test;
% Kxx = E_Kxx;
% 
% m_sum = zeros(Q,1);
% for k = 1:N_test
%     x = x_test(:,k);
%     m_test = g(x);
%     m_sum = m_sum+m_test;
% end
% m_sum = m_sum/N_test;
% C_test = zeros(D,Q);
% C_sum = zeros(D,Q);
% for k = 1:N_test
%     Cgx = zeros(D,Q);
%     x = x_test(:,k);
%     for q = 1:Q
%         Exq = zeros(D,N*Q);
%         for n = 1:N
%             for p = 1:Q
%                 x_np = x_sigma(:,(n-1)*Q+p);
%                 kxq = zeros(D,1);
%                 for e = 1:E
%                     kxq = kxq + weights(e,q)*weights(e,p)*covfunc(GPs(e).hyp.cov,x',x_np')*x;
%                 end
%                 Exq(:,(n-1)*Q+p) = kxq;
%             end
%         end
%         C_test(:,q) = Exq*Kc';
%     end
%     C_test = C_test - m*mu';
%     C_sum = C_sum + C_test;
% end
% C_mean = C_sum/N_test;

% C_gg_sum = zeros(Q,Q);
% CxEg_sum = zeros(Q,Q);
% eig_sum = zeros(Q,1);
% eigC_sum = zeros(Q,1);
% mineig = zeros(1,N_test);
% mineigC = zeros(1,N_test);
% for k = 1:N_test
%     kx = zeros(Q,N*Q);
%     x = x_test(:,k);
%     for q = 1:Q
%         for n = 1:N
%             for p = 1:Q
%                 x_np = x_sigma(:,(n-1)*Q+p);
%                 kpq = 0;
%                 for e = 1:E
%                     kpq = kpq + weights(e,p)*weights(e,q)*covfunc(GPs(e).hyp.cov,x',x_np');
%                 end
%                 kx(q,(n-1)*Q+p) = kpq;
%             end
%         end
%     end
%     CxEg = kx*Kv1*kx' - mu*mu';
%     CxEg_sum = CxEg_sum + CxEg;
%     eigCxEg = eig(CxEg);
%     mineig(k) = min(eigCxEg);
%     eig_sum = eig_sum + eigCxEg;
%     Cgg = Kxx - kx*Kx_inv*kx';
%     C_gg_sum = C_gg_sum + Cgg;
%     eigC = eig(Cgg);
%     mineigC(k) = min(eigC);
%     eigC_sum = eigC_sum + eigC;
%     disp(k)
% end
% CxEg_mean = CxEg_sum/N_test;
% eig_mean = eig_sum/N_test;
% C_gg_mean = C_gg_sum/N_test;
% eigC_mean = eigC_sum/N_test;
