func = @(x) x.*cos(x);

x_train = getSigmaPoints(1,'UKF');
y_train = func(x_train);
x_test = getSigmaPoints(1,'Random');
x_test = sort(x_test);

hyp = [log(60) log(1)];
Kxx = covSEard(hyp, x_train');
Kxxinv = inv(Kxx);
Ksx = covSEard(hyp, x_test', x_train');
Kxs = Ksx';
Kss = covSEard(hyp, x_test');
mu = Ksx*Kxxinv*y_train';
S = Kss - Ksx*Kxxinv*Kxs;
s = diag(S);


