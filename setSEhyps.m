function [l,alpha] = setSEhyps(E,D,pattern)

l_base = [4, 2];
if strcmp(pattern,'mo')
%     l = [2.4, 1.5; 2.4, 1.5];  % l should be E x D
%     alpha = [1; 1];  % alpha should be E x 1
    
    % l_base = [4, 2];
    alpha_base = 1;
    l = repmat(l_base,[E,1]);
    alpha = repmat(alpha_base, [E,1]);
elseif strcmp(pattern,'so')
    l = repmat(l_base,[E,1]);  % l should be E x D
    alpha = [1; 1];  % alpha should be E x 1
else
    error('Pattern error.')
end

if ~isequal(size(l),[E,D]) || ~isequal(size(alpha),[E,1])
    error('Multi gps hyps setting error.')
end
end