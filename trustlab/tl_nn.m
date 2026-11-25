function [s, q] = tl_nn(X, Ub, Ua, c)
[Xu, ~, g] = unique(X, 'rows');
w = ones(1, size(X, 2));
dB = tl_knn(Xu, Ub, w, 1, c);
dA = tl_knn(Xu, Ua, w, 1, c);
sB = sqrt(dB(g)); sA = sqrt(dA(g));
s = sB ./ (sA + sB); s(sA + sB == 0) = 0.5;
q = sB / (1023 * sqrt(size(X, 2)));
end
