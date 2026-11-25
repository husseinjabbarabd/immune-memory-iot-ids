function r = tl_osr(Tr, yTr, Vb, T, ben, nmax, far, seed, c)
rs = 0.01 * 1023;
Tr = double(Tr); Vb = double(Vb); T = double(T);
Ub = unique(Tr(yTr == ben, :), 'rows'); Ua = unique(Tr(yTr ~= ben, :), 'rows');
[C, R] = nsa(Ua, Ub, nmax, rs, seed(1), c);
mV = score(Vb, C, R); mT = score(T, C, R);
nb = sum(yTr == ben);
K = tl_kmeans(Tr(yTr == ben, :), min(8, max(2, floor(nb / 500))), seed(2), c);
d95 = prctile(near(Tr(yTr == ben, :), K), 95);
kV = near(Vb, K) / d95; kT = near(T, K) / d95;
r.detectors = size(C, 1); r.max_detectors = nmax; r.nsa_V = mV; r.nsa_T = mT; r.n09_V = kV; r.n09_T = kT;
r.flag = {'nsa_native', mT >= 0, mean(mV >= 0); 'n09_native', kT > 1, mean(kV > 1)};
for o = 1:3
    t = tl_match(mV, far(o)); r.flag(end+1, :) = {sprintf('nsa%d', o), mT >= t, mean(mV >= t)};
    t = tl_match(kV, far(o)); r.flag(end+1, :) = {sprintf('n09_%d', o), kT >= t, mean(kV >= t)};
end
end

function [C, R] = nsa(Ua, Ub, nmax, rs, seed, c)
o = randperm(RandStream('twister', 'Seed', seed), size(Ua, 1));
C = zeros(0, size(Ua, 2)); R = zeros(0, 1); nb = 40000; w = ones(1, size(Ua, 2));
for a = 1:nb:numel(o)
    if size(C, 1) >= nmax; break; end
    X = Ua(o(a:min(a + nb - 1, end)), :);
    if ~isempty(C); X = X(~covered(X, C, R), :); end
    if isempty(X); continue; end
    d = sqrt(tl_knn(X, Ub, w, 1, c)) - rs;
    X = X(d > rs, :); d = d(d > rs);
    for i = 1:size(X, 1)
        if size(C, 1) >= nmax; break; end
        if ~any(sqrt(sum((C - X(i, :)).^2, 2)) <= R); C(end+1, :) = X(i, :); R(end+1, 1) = d(i); end
    end
end
end

function v = covered(X, C, R)
v = false(size(X, 1), 1); nb = max(1, floor(1e7 / size(C, 1)));
for a = 1:nb:size(X, 1)
    b = a:min(a + nb - 1, size(X, 1));
    v(b) = any(dist(X(b, :), C) <= R', 2);
end
end

function m = score(X, C, R)
n = size(X, 1); nb = max(1, floor(1e7 / size(C, 1))); nk = ceil(n / nb); M = cell(nk, 1); Rt = R';
parfor i = 1:nk
    r = (i - 1) * nb + 1:min(i * nb, n);
    M{i} = max(Rt - dist(X(r, :), C), [], 2);
end
m = vertcat(M{:});
end

function d = near(X, C)
d = min(dist(X, C), [], 2);
end

function D = dist(X, C)
D = sqrt(max(0, sum(X.^2, 2) + sum(C.^2, 2)' - 2 * X * C'));
end
