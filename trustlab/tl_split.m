function sp = tl_split(y, sp2, c)
rs = RandStream('twister', 'Seed', c.seed_split);
sp = zeros(numel(y), 1, 'uint8');
for k = 1:max(y)
    idx = find(y == k); n = numel(idx);
    nTe = round(c.test_frac * n);
    pool = idx(sp2(idx) == 1);
    te = pool(randperm(rs, numel(pool), nTe));
    rest = setdiff(idx, te);
    va = rest(randperm(rs, numel(rest), round(c.val_frac * (n - nTe))));
    sp(idx) = 1; sp(te) = 3; sp(va) = 2;
end
end
