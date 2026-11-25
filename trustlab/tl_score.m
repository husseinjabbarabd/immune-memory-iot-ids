function [s, q, dA, dB] = tl_score(X, cells, isB, block)
X = double(X); C = double(cells);
n = size(X, 1); xn = sum(X.^2, 2); cn = sum(C.^2, 2)'; Ct = C';
iB = find(isB); iA = find(~isB); nb = max(1, floor(block / size(C, 1)));
dB = zeros(n, 1); dA = zeros(n, 1);
for a = 1:nb:n
    e = min(a + nb - 1, n);
    D = round(xn(a:e) + cn - 2 * (X(a:e, :) * Ct));
    dB(a:e) = min(D(:, iB), [], 2); dA(a:e) = min(D(:, iA), [], 2);
end
sB = sqrt(dB); sA = sqrt(dA);
s = sB ./ (sA + sB); s(sA + sB == 0) = 0.5;
q = sB / (1023 * sqrt(size(X, 2)));
end
