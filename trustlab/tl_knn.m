function [Dk, Ik] = tl_knn(X, ref, w, K, c)
n = size(X, 1); N = size(ref, 1); X = double(X);
Xw = X .* w; xnw = (X.^2) * w';
nb = 5000; nq = ceil(n / nb); out = cell(nq, 1);
rc = c.refchunk; qb = c.qblock;
parfor i = 1:nq
    q = (i - 1) * nb + 1:min(i * nb, n);
    key = inf(numel(q), K);
    for r0 = 1:rc:N
        r1 = min(r0 + rc - 1, N);
        R = double(ref(r0:r1, :)); rn = sum(R.^2 .* w, 2)'; gi = (r0 - 1):(r1 - 1);
        for a2 = 1:qb:numel(q)
            e = min(a2 + qb - 1, numel(q)); qq = q(a2:e);
            D = round(xnw(qq) + rn - 2 * (Xw(qq, :) * R'));
            key(a2:e, :) = mink([key(a2:e, :), mink(D * N + gi, K, 2)], K, 2);
        end
    end
    out{i} = key;
end
key = vertcat(out{:});
Dk = floor(key / N); Ik = key - Dk * N + 1;
end
