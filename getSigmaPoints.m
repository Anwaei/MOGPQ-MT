function [Xi_s, N] = getSigmaPoints(D,name)
% Get the unit sigma points

if strcmp(name, 'UKF')
    samplings.name = 'UKF';
    samplings.sampling = GaussianSamplingUKF();
    samplings.sampling.setSampleScaling(1);
elseif strcmp(name, '5th-Degree CKF')
    samplings.name = '5th-Degree CKF';
    samplings.sampling = GaussianSamplingCKF();
elseif strcmp(name, 'Symmetric LCD')
    samplings.name = 'Symmetric LCD';
    samplings.sampling = GaussianSamplingLCD();
    samplings.sampling.setNumSamplesByFactor(10);
elseif strcmp(name, 'Asymmetric LCD')
    samplings.name = 'Asymmetric LCD';
    samplings.sampling = GaussianSamplingLCD();
    samplings.sampling.setSymmetricMode(false);
    samplings.sampling.setNumSamples(15);
elseif strcmp(name, 'RUKF')
    samplings.name = 'RUKF';
    samplings.sampling = GaussianSamplingRUKF();
    samplings.sampling.setNumIterations(10);
elseif strcmp(name, 'Gauss-Hermite')
    samplings.name = 'Gauss-Hermite';
    samplings.sampling = GaussianSamplingGHQ();
    samplings.sampling.setNumQuadraturePoints(3);
elseif strcmp(name, 'Random')
    samplings.name = 'Random';
    samplings.sampling = GaussianSamplingRnd();
    samplings.sampling.setNumSamples(50); 
else
    error('Please select a vaild sampling method.')
end

[samples, weights, numSamples] = samplings.sampling.getStdNormalSamples(D);

Xi_s = samples;
N = numSamples;

end