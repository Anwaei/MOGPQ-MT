syms y1 y2 k11 k12 k22 e11 e12 e22;
y = [y1;y2];
K = [k11,k12;k12,k22];
E = [e11,e12;e12,e22];
y.'*K*E*K*y