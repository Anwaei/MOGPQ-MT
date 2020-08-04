% Solve the expectations for LMC settings with SE cov GPs
function [E1, E2, E3, E4] = solveExpectations(m, P, xi, conf_mo)

Q = conf_mo.Q; D = conf_mo.D; E = conf_mo.LMCsettings.E; 
[D1, NQ] = size(xi); N = NQ/Q;
if D~=D1
    error('Input dimension error');
end 
GPs =  conf_mo.LMCsettings.gp;
weights = conf_mo.LMCsettings.weights;
L = chol(P, 'lower');

%% solve Me, Ce, Cee'
M = zeros(D,D,E);
M_inv = zeros(D,D,E);
Im = eye(D);
C1 = zeros(E,1);
C2 = zeros(E,E);
alpha = zeros(E,1);
for e = 1:E
    Lambda = diag(exp(GPs(e).hyp.cov(1:D))).^2;
    alphae = exp(GPs(e).hyp.cov(D+1));
    Me_inv = L'/Lambda*L;
    Me = Im/Me_inv;
    M(:,:,e) = Me;
    M_inv(:,:,e) = Me_inv;
    alpha(e) = alphae;
    C1(e) = alphae^2*det(Me_inv+Im)^(-1/2);  % Why M_inv(1) and M_inv(3) so analogous?
end
for e = 1:E
    for f = 1:E
        C2(e,f) = alpha(e)^2*alpha(f)^2*det(M_inv(:,:,e)+M_inv(:,:,f)+Im)^(-1/2);
    end
end
    
%% solve E1 = Ex[K(x,x)] = Eu[K(m+Lu,m+Lu)]
E1 = zeros(Q,Q);
for p = 1:Q
    for q = 1:Q
        E_kpqxx = 0;
        for e = 1:E
            E_kpqxx = E_kpqxx + weights(e,p)*weights(e,q)*alpha(e)^2;
        end
        E1(p,q) = E_kpqxx;
    end
end
%% solve E2 = Ex[K(x)] = Eu[K(m+Lu)]
E2 = zeros(Q,N*Q);
for q = 1:Q
    for n = 1:N
        for p = 1:Q
            xi_np = xi(:,(n-1)*Q+p);
            E_kpq = 0;
            for e = 1:E
                E_ke = C1(e)*exp(-1/2*xi_np'/(M(:,:,e)+Im)*xi_np);
                E_kpq = E_kpq + weights(e,p)*weights(e,q)*E_ke;
            end
            E2(q,(n-1)*Q+p) = E_kpq;
        end
    end
end
%% solve E3 E3(:,:,p,q) = Ex[kq(x)'kp(x)] = Eu[kq(m+Lu)'kp(m+Lu)]
E3 = zeros(N*Q,N*Q,Q,Q);
for p = 1:Q
    for q = 1:Q
        E_kqkp = zeros(N*Q,N*Q);
        for m = 1:N
            for n = 1:N
                for s = 1:Q
                    for t = 1:Q
                        xi_ms = xi(:,(m-1)*Q+s);
                        xi_nt = xi(:,(n-1)*Q+t);
                        E_kqskpt = 0;
                        E_test = zeros(E,E);
                        for e = 1:E
                            for f = 1:E
                                z = M_inv(:,:,e)*xi_ms + M_inv(:,:,f)*xi_nt;
                                E_kekf = C2(e,f) * exp(-1/2*(xi_ms'*M_inv(:,:,e)*xi_ms +...
                                    xi_nt'*M_inv(:,:,f)*xi_nt - z'/(M_inv(:,:,e) + ...
                                    M_inv(:,:,f) + Im)*z));
                                E_kqskpt = E_kqskpt + weights(e,q)*weights(e,s)...
                                    *weights(f,p)*weights(f,t)*E_kekf;
                                E_test(e,f) = weights(e,q)*weights(e,s)...
                                    *weights(f,p)*weights(f,t)*E_kekf;
%                                 % ------------ test ------------
                            end
                        end
                        E_kqkp((m-1)*Q+s,(n-1)*Q+t) = E_kqskpt;
                    end
                end
            end
        end
        E3(:,:,p,q) = E_kqkp;
    end
end
                
%% sovle E4 E4(:,:,q) = Eu[ukq(m+Lu)]
E4 = zeros(D,N*Q,Q);
for q = 1:Q
    E_xkq = zeros(D,N*Q);
    for n = 1:N
        for p = 1:Q
            xi_np = xi(:,(n-1)*Q+p);
            E_kqpx = 0;
            for e = 1:E
                MeIinv = Im/(M(:,:,e)+Im);
                E_kex = C1(e)*exp(-1/2*(xi_np'*MeIinv*xi_np)) *MeIinv *xi_np;
                E_kqpx = E_kqpx + weights(e,q)*weights(e,p)*E_kex;
            end
            E_xkq(:,(n-1)*Q+p) = E_kqpx;
        end
    end
    E4(:,:,q) = E_xkq;            
end

% test
%                                 xi_test = Gaussian(zeros(D,1),eye(D));
%                                 kekf_test_sum = 0;
%                                 for num = 1:10000
%                                     xi_sample = xi_test.drawRndSamples(1);
%                                     x_sample = m + L*xi_sample; x_ms = m + L*xi_ms; x_nt = m + L*xi_nt;
%                                     ke_test = GPs(e).covfunc(GPs(e).hyp.cov,x_sample',x_ms');
%                                     
%                                     
% %                                     alphae_test = exp(GPs(e).hyp.cov(D+1));
% %                                     Lambda_test = diag(exp(GPs(e).hyp.cov(1:D))).^2;
% %                                     ke_test_test = alphae_test*exp(-1/2*((x_sample-x_ms)'*inv(Lambda_test)*(x_sample-x_ms)));
%                                     
%                                     kf_test = GPs(f).covfunc(GPs(f).hyp.cov,x_sample',x_nt');
%                                     kekf_test = ke_test*kf_test;
%                                     kekf_test_sum = kekf_test_sum + kekf_test;
%                                 end
%                                 kekf_test_mean = kekf_test_sum/num;
%                                 err = E_kekf - kekf_test_mean;
%                                 disp(err);