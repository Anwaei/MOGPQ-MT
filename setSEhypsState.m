function [l,alpha] = setSEhypsState(E,D)


l = [80 12 1; 80 12 1; 80 12 1];  % l should be E x D
% l = [10 10 10; 10 10 10; 10 10 10];
alpha = [1; 1; 1];  % alpha should be E x 1
% alpha = [0.5; 0.5; 0.5];
if ~isequal(size(l),[E,D]) || ~isequal(size(alpha),[E,1])
    error('Multi gps hyps setting error.')
end
end