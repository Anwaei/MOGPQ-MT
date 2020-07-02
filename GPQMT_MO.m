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

[E1, E2, E3, E4] = solveExpectations(m,P,xi_sigma,conf_mo);

Kx = createGramMatrix(x_sigma, conf_mo); 
I = eye(size(Kx));Kx_inv = I/Kx;
Kv = Kx_inv*(y*y')*Kx_inv -Kx_inv;
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

E_K1 = zeros(Q,Q);
for p = 1:Q
    for q = 1:Q
        E_K1(p,q) = trace(E3(:,:,p,q)*Kx_inv);
    end
end


mu = E_Kx * Kx_inv * y;
Pi = E_Kv + E_Kxx - mu*mu';
C = L * E_Kc;

end