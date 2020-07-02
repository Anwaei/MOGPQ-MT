function [data_train, conf_mo] = generateTrainingData(m,P,g,conf_mo)
Q = conf_mo.Q; D = conf_mo.D;
% get training inputs
sample_method = 'UKF';
[~, N] = getSigmaPoints(D, sample_method);
conf_mo.N = N;
xi_sigma = zeros(D,N,Q);
for q = 1:Q
    xi_sigma(:,:,q) = getSigmaPoints(D, sample_method);
end
xi_sigma = reshape(permute(xi_sigma,[1 3 2]),D,N*Q);
L = chol(P, 'lower');
x_sigma = m + L*xi_sigma;
% get training outputs
obs_noise = conf_mo.obs_noise;
noise = obs_noise.drawRndSamples(1);
if ~isequal(size(noise),[Q,1])
    error('Noise dimension setting error');
end
y_train = zeros(N*Q,1);
for n = 1:N
    for q = 1:Q
        y = g(x_sigma(:,(n-1)*Q+q)) + obs_noise.drawRndSamples(1);
        y_train((n-1)*Q+q) = y(q);
    end
end
data_train.xi_sigma = xi_sigma;
data_train.x_sigma = x_sigma;
data_train.y_train = y_train;
end