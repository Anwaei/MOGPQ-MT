function [l,alpha] = setSEhyps(E,D,pattern)

if strcmp(pattern,'mo')
    l = [2.4, 1.5; 2.4, 1.5];  % l should be E x D
    % l = [1.3 1.2; 1.2 1.5];
    alpha = [1; 1];  % alpha should be E x 1
elseif strcmp(pattern,'so')
    l = [2.4, 1.5; 2.4, 1.5];  % l should be E x D
    alpha = [1; 1];  % alpha should be E x 1
else
    error('Pattern error.')
end

if ~isequal(size(l),[E,D]) || ~isequal(size(alpha),[E,1])
    error('Multi gps hyps setting error.')
end
end