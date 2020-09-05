function [mu, Pi] = CooUT(m, P, func_g, conf_mo)

sampling = GaussianSamplingUKF();
sampling.setSampleScaling(2);
gaussian = Gaussian(m, P);
[x_sigma, weights, numSamples] = sampling.getSamples(gaussian);

y_sigma = func_g(x_sigma);
mu = zeros(conf_mo.Q, 1);
Pi = zeros(conf_mo.Q, conf_mo.Q);

for i = 1:numSamples
    mu = mu + weights(i)*y_sigma(:,i);
end
for i = 1:numSamples
    Pi = Pi + weights(i)*(y_sigma(:,i)-mu)*(y_sigma(:,i)-mu)';
end

end