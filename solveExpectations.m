% Solve the expectations for LMC settings with SE cov GPs
function [E1, E2, E3, E4] = solveExpectations(m, P, xi, conf_mo)

Q = conf_mo.Q; D = conf_mo.D; E = conf_mo.LMCsettings.E;
GPs =  conf_mo.LMCsettings.gp;
weights = conf_mo.LMCsettings.weights;
%% solve E1 = Ex[K(x,x)] = Eu[K(m+Lu,m+Lu)]
E1 = zeros(Q,Q);
for p = 1:Q
    for q = 1:Q
        E_kpqxx = 0;
        for e = 1:E
            alpha = exp(GPs(e).hyp.cov(D+1));
            E_kpqxx = E_kpqxx + weights(e,p)*weights(e,q)*alpha^2;
        end
        E1(p,q) = E_kpqxx;
    end
end
%% solve E2 = Ex[K(x)] = Eu[K(m+Lu)]

%% solve E3 E3(:,:,p,q) = Ex[kq(x)'kp(x)] = Eu[kq(m+Lu)'kp(m+Lu)]

%% sovle E4 E4(:,:,q) = Eu[ukq(m+Lu)]

end