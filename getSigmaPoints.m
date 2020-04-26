function [Xi_s, N] = getSigmaPoints(D)
% Get the unit sigma points

samplings{1}.name = 'UKF';
samplings{1}.sampling = GaussianSamplingUKF();
samplings{1}.sampling.setSampleScaling(1);

[samples, weights, numSamples] = samplings{1}.sampling.getStdNormalSamples(D);

Xi_s = samples;
N = numSamples;

end