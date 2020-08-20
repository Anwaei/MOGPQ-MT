function [l,alpha] = setSEhypsMeas(E,D)

l = [80 5 0.5];  % l should be E x D
alpha = 1;  % alpha should be E x 1
if ~isequal(size(l),[E,D]) || ~isequal(size(alpha),[E,1])
    error('Multi gps hyps setting error.')
end
end
