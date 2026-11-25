function mi = tl_mi(C, y)
n = numel(y); nC = max(y); mi = zeros(1, size(C, 2));
py = accumarray(y(:), 1, [nC 1])' / n;
for j = 1:size(C, 2)
    P = accumarray([double(C(:, j)) + 1, y(:)], 1, [1024 nC]) / n;
    px = sum(P, 2); E = px * py; k = P > 0;
    mi(j) = sum(P(k) .* log2(P(k) ./ E(k)));
end
end
