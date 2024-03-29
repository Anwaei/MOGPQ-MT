function [l,alpha] = setSEhypsState(E,D,pattern)

if strcmp(pattern,'mo')
    l = [90 45 3.8; 75 60 2.4; 90 70 3.9];  % l should be E x D
    alpha = [1; 1; 1];  % alpha should be E x 1
elseif strcmp(pattern,'so')
    l = kron([90 55 3.7], ones(D,1));  % l should be E x D
    alpha = [1; 1; 1];  % alpha should be E x 1
else
    error('Pattern error.')
end

if ~isequal(size(l),[E,D]) || ~isequal(size(alpha),[E,1])
    error('Multi gps hyps setting error.')
end
end