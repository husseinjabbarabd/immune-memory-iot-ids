function [cells, lab, AT] = tl_airs2(code, y, cls, c)
code = double(code); n = size(code, 1); d = size(code, 2); dmax = sqrt(d);
rs = RandStream('twister', 'Seed', c.seed_airs);
i1 = randi(rs, n, c.npairs, 1); i2 = randi(rs, n, c.npairs, 1);
kp = i1 ~= i2; i1 = i1(kp); i2 = i2(kp); a = 0;
for s = 1:100000:numel(i1)
    e = min(s + 99999, numel(i1));
    a = a + sum(sqrt(sum((code(i1(s:e), :) / 1023 - code(i2(s:e), :) / 1023).^2, 2)) / dmax);
end
AT = a / numel(i1);
A = cell(numel(cls), 1);
for j = 1:numel(cls)
    rso = RandStream('twister', 'Seed', c.seed_airs + cls(j));
    rows = find(y == cls(j));
    A{j} = code(rows(randperm(rso, numel(rows))), :)';
end
M = cell(numel(cls), 1); P = c.airs; sm = c.seed_mut;
parfor j = 1:numel(cls)
    M{j} = airs2(A{j}, P, AT, RandStream('twister', 'Seed', sm + cls(j)));
end
cells = zeros(0, d); lab = zeros(0, 1);
for j = 1:numel(cls)
    cells = [cells; M{j}]; lab = [lab; cls(j) * ones(size(M{j}, 1), 1)];
end
end

function MC = airs2(A, P, AT, rs)
[d, n] = size(A); dmax = sqrt(d); S = 1023;
cap = 4096; T = zeros(d, cap); Tn = inf(1, cap);
m = min(P.SEED_CELLS, n); T(:, 1:m) = A(:, 1:m); Tn(1:m) = sum(T(:, 1:m).^2, 1);
thr = AT * P.ATS;
for i = m+1:n
    ai = A(:, i)';
    [~, b] = min(Tn + (ai * ai') - 2 * (ai * T));
    ag = ai / S; best = T(:, b)' / S;
    sb = 1 - sqrt(sum((best - ag).^2)) / dmax;
    pool = [best; mut(repmat(best, round(sb * P.CLONE_RATE * P.MUTATE_RATE), 1), 1 - sb, rs)];
    it = 0;
    while true
        it = it + 1;
        s = 1 - sqrt(sum((pool - ag).^2, 2)) / dmax;
        [sc, ic] = max(s); cand = pool(ic, :);
        if mean(s) >= P.STIM_THRESH || it >= P.MAX_ARB_ITER; break; end
        r = s * P.CLONE_RATE;
        [r, o] = sort(r, 'ascend'); pool = pool(o, :); s = s(o);
        tot = sum(r); k = 1;
        while tot > P.TOTAL_RES
            ex = tot - P.TOTAL_RES;
            if r(k) <= ex; tot = tot - r(k); k = k + 1; else; r(k) = r(k) - ex; tot = P.TOTAL_RES; end
        end
        pool = pool(k:end, :); s = s(k:end);
        par = repelem((1:size(pool, 1))', round(s * P.CLONE_RATE));
        if ~isempty(par); pool = [pool; mut(pool(par, :), 1 - s(par), rs)]; end
    end
    if sc > sb
        m = m + 1;
        if m > cap; T(:, cap+1:2*cap) = 0; Tn(cap+1:2*cap) = inf; cap = 2 * cap; end
        ci = round(cand * S); T(:, m) = ci'; Tn(m) = ci * ci';
        if sqrt(sum((cand - best).^2)) / dmax < thr
            T(:, b) = T(:, m); Tn(b) = Tn(m); T(:, m) = 0; Tn(m) = inf; m = m - 1;
        end
    end
end
MC = T(:, 1:m)';
end

function R = mut(R, r, rs)
lo = max(R - r / 2, 0); hi = min(R + r / 2, 1);
R = round((lo + rand(rs, size(R)) .* (hi - lo)) * 1023) / 1023;
end
