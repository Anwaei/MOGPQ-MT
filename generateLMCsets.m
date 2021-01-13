function confs = generateLMCsets(maxE)

conf_mo.D = 2;  % num input
conf_mo.Q = 2;  % num output
obs_noise = Gaussian([0;0],[1e-10,0;0,1e-10]);
conf_mo.obs_noise = obs_noise;  % observation noise
conf_mo.model = 'LMC';
conf_mo.sample_cov_ref = eye(conf_mo.Q);
conf_mo.sample_method = 'UKF';
confs = cell(1,maxE);

weights_sets = cell(1,maxE);
weights_sets{1} = 1;
weights_sets{2} = [1, 0.2; 0.2, 1];
weights_sets{3} = [1, 0.2; 0.2, 1];
weights_sets{4} = 1;
weights_sets{5} = 1;

for E = 1:maxE
    conf_mo = rmfield(conf_mo,'LMCsettings');
    conf_mo.LMCsettings.E = E;  % num latent functions
    conf_mo.LMCsettings.weights = weights_sets{E};
    conf_mo.LMCsettings.gp = struct('covfunc',cell(E,1),'meanfunc',cell(E,1),'hyp',cell(E,1));
    [l,alpha] = setSEhyps(E,conf_mo.D, 'mo');
    for e = 1:E  % set each gp
        conf_mo.LMCsettings.gp(e).covfunc = @covSEard;
        conf_mo.LMCsettings.gp(e).meanfunc = [];
        conf_mo.LMCsettings.gp(e).hyp.cov = [log(l(e,:)) log(alpha(e,:))];
        conf_mo.LMCsettings.gp(e).hyp.lik = log(sqrt(0.4));
    end
    confs{E} = conf_mo;
end

end