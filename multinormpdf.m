function p = multinormpdf(x, m, P)

D = numel(x);
t = -1/2 * (x-m)' /P * (x-m);
p = (1/(2*pi)^(D/2)) * (1/sqrt(det(P))) * exp(t);

end