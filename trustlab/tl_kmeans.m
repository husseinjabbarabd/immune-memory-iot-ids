function [C, it] = tl_kmeans(X, k, seed, c)
[U, ~, g] = unique(double(X), 'rows'); n = accumarray(g, 1);
it = 0;
if size(U, 1) <= k; C = U; return; end
rs = RandStream('twister', 'Seed', seed);
C = zeros(k, size(U, 2));
C(1, :) = U(randsample(rs, size(U, 1), 1, true, n), :);
d = sum((U - C(1, :)).^2, 2);
for j = 2:k
    C(j, :) = U(randsample(rs, size(U, 1), 1, true, n .* d), :);
    d = min(d, sum((U - C(j, :)).^2, 2));
end
K = parallel.pool.Constant(U);
a = zeros(size(U, 1), 1);
for it = 1:c.km_iter
    a0 = a; a = near(K, C, size(U, 1), c);
    m = accumarray(a, n, [k 1]); ok = m > 0;
    for j = 1:size(U, 2)
        sj = accumarray(a, n .* U(:, j), [k 1]); C(ok, j) = sj(ok) ./ m(ok);
    end
    if isequal(a, a0); break; end
end
C = round(C);
end

function a = near(K, C, n, c)
nb = max(1, floor(c.block / size(C, 1))); nk = ceil(n / nb); A = cell(nk, 1);
Ct = C'; cn = sum(C.^2, 2)';
parfor i = 1:nk
    U = K.Value;
    r = (i - 1) * nb + 1:min(i * nb, n);
    [~, A{i}] = min(sum(U(r, :).^2, 2) + cn - 2 * U(r, :) * Ct, [], 2);
end
a = vertcat(A{:});
end
