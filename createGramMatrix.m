function K = createGramMatrix(x,conf_mo)
Q = conf_mo.Q; E = conf_mo.LMCsettings.E; 
[~, NQ] = size(x); N = NQ/Q;
GPs =  conf_mo.LMCsettings.gp;
weights = conf_mo.LMCsettings.weights;
K = zeros(NQ,NQ);
for m = 1:N
    for n = 1:N
        for p = 1:Q
            for q = 1:Q
                k = 0;
                x_mp = x(:,(m-1)*Q+p);
                x_nq = x(:,(n-1)*Q+q);
                for e = 1:E
                    k = k + weights(e,p)*weights(e,q)*GPs(e).covfunc(GPs(e).hyp.cov, x_mp',x_nq');
                end
                K((m-1)*Q+p,(n-1)*Q+q) = k;
            end
        end
    end
end
end