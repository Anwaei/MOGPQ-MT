function p = multinormpdf2(x1, x2, m, P)

m1 = m(1);
m2 = m(2);
t1 = x1-m1;
t2 = x2-m2;
Pinv = inv(P);
p11 = Pinv(1,1);
p12 = Pinv(1,2);
p21 = Pinv(2,1);
p22 = Pinv(2,2);
D = 2;

% t = -1/2 * (x-m)' /P * (x-m);
t = -1/2 * (t1.^2.*p11 + t1.*t2.*p21 + t2.*t1.*p12 + t2.^2.*p22);
p = (1/(2*pi)^(D/2)) * (1/sqrt(det(P))) .* exp(t);

end