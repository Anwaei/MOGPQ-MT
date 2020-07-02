function [l,alpha] = setSEhyps(E,D)

l = [2 1.5; 1.9, 1.6];  % l should be E x D
alpha = [1; 1];  % alpha should be E x 1
if ~isequal(size(l),[E,D]) || ~isequal(size(alpha),[E,1])
    error('Multi gps hyps setting error.')
end
end