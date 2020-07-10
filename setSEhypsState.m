function [l,alpha] = setSEhypsState(E,D)

l = [12 2 1; 15 2 2; 12 1 1];  % l should be E x D
alpha = [1; 1; 1];  % alpha should be E x 1
if ~isequal(size(l),[E,D]) || ~isequal(size(alpha),[E,1])
    error('Multi gps hyps setting error.')
end
end