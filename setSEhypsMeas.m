function [l,alpha] = setSEhypsMeas(E,D,pattern)

if strcmp(pattern,'mo')
    l = [80 15 0.8];  % l should be E x D
    alpha = 1;  % alpha should be E x 1
elseif strcmp(pattern,'so')
    l = [80 15 0.8];  % l should be E x D
    alpha = 1;  % alpha should be E x 1
else
    error('Pattern error.')
end

if ~isequal(size(l),[E,D]) || ~isequal(size(alpha),[E,1])
    error('Multi gps hyps setting error.')
end
end
