function run_nested(tag, drop, published)
if nargin < 2; drop = strings(0); end
if nargin < 3; published = false; end
c = tl_cfg();
od = fullfile(c.res, tag);
if ~exist(od, 'dir'); mkdir(od); end
tl_pool(c);

d = tl_data(c); X = d.X; y = d.y; sp = d.sp; fn = d.fn; names = d.names; ben = d.ben; clear d
cand = find(~ismember(fn, drop))'; F = numel(cand);
assert(F == numel(fn) - numel(drop));
te = find(sp == 3); yTe = y(te);
att = setdiff(1:numel(names), ben);
rounds = [num2cell(att), {find(ismember(names, c.twins))'}];
O = cell(numel(rounds), 1);
if published
    assert(isempty(drop), 'published selections exist for the nested rounds only');
    P = readtable(fullfile(c.ref, 'round_outputs.csv'), 'TextType', 'string', 'VariableNamingRule', 'preserve');
    assert(height(P) == numel(rounds));
end

for r = 1:numel(rounds)
    x = rounds{r}; rn = strjoin(names(x), '+');
    cls = setdiff(1:numel(names), x);
    tr = find(sp == 1 & ~ismember(y, x)); va = find(sp == 2 & ~ismember(y, x));
    yTr = y(tr); yVa = y(va);
    m = tl_map(X(tr, cand));
    cTr = tl_enc(X(tr, cand), m); cVa = tl_enc(X(va, cand), m);

    if published
        assert(P.round(r) == r && P.family(r) == rn);
        G = struct('k', P.m(r), 'wsel', pick(P.('features (weight)')(r), fn(cand)), 'klist', [], 'f', []);
    else
        D = tl_frame(cTr, yTr, cls, ben, c);
        K = parallel.pool.Constant(D); c2 = c; c2.klist(1) = F;
        G = tl_clonalg(K, F, c2); clear K D
    end
    fi = find(G.wsel > 0);

    [cells, lab, AT] = tl_airs2(cTr(:, fi), yTr, cls, c);
    rd = fullfile(od, sprintf('r%02d', r)); if ~exist(rd, 'dir'); mkdir(rd); end
    save(fullfile(rd, 'airs2.mat'), 'cells', 'lab', 'AT', '-v7');
    isB = lab == ben;
    [sV, qV, dAv, dBv] = tl_score(cVa(:, fi), cells, isB, c.block);
    tl_dist(fullfile(rd, 'score_val.mat'), dAv, dBv);
    taus = tl_tau(sV, yVa ~= ben, c);
    vb = yVa == ben; sVb = sV(vb); qVb = qV(vb);
    nu = quantile(qVb, 1 - c.alpha);

    cTe = tl_enc(X(te, cand(fi)), m, fi);
    [s, q, dA, dB] = tl_score(cTe, cells, isB, c.block);
    tl_dist(fullfile(rd, 'score_test.mat'), dA, dB);
    isX = ismember(yTe, x); isBt = yTe == ben; isK = ~isX & ~isBt;

    o = struct('round', r, 'family', rn, 'k', G.k, 'feat', strjoin(fn(cand(fi))', ';'), ...
        'w', mat2str(G.wsel(fi)), 'cells', size(cells, 1), 'AT', AT, 'nu', nu, ...
        'tieVal', sum(dAv == 0 & dBv == 0), 'tieTest', sum(dA == 0 & dB == 0), 'nX', sum(isX));
    for j = 1:3
        tau = taus(j);
        farV = mean(sVb >= tau | qVb >= nu);
        tM = tl_match(sVb, farV); nM = tl_match(qVb, farV);
        R = {s >= tau, s >= tau | q >= nu, s >= tM, q >= nM};
        V = {sVb >= tau, sVb >= tau | qVb >= nu, sVb >= tM, qVb >= nM};
        o.(sprintf('tau%d', j)) = tau; o.(sprintf('tauM%d', j)) = tM; o.(sprintf('nuM%d', j)) = nM;
        rl = 'aocq';
        for i = 1:4
            o.(sprintf('det_%s%d', rl(i), j)) = mean(R{i}(isX));
            o.(sprintf('far_%s%d', rl(i), j)) = mean(R{i}(isBt));
            o.(sprintf('kno_%s%d', rl(i), j)) = mean(R{i}(isK));
            o.(sprintf('vfar_%s%d', rl(i), j)) = mean(V{i});
        end
        if numel(x) > 1
            for z = x
                o.(sprintf('det_%s_%d', names(z), j)) = mean(s(yTe == z) >= tau | q(yTe == z) >= nu);
                o.(sprintf('detA_%s_%d', names(z), j)) = mean(s(yTe == z) >= tau);
                o.(sprintf('detC_%s_%d', names(z), j)) = mean(s(yTe == z) >= tM);
                o.(sprintf('detQ_%s_%d', names(z), j)) = mean(q(yTe == z) >= nM);
            end
        end
    end
    sx = single(s); qx = single(q); w = G.wsel; klist = G.klist; fcurve = G.f;
    save(fullfile(od, sprintf('round_%02d.mat', r)), 'o', 'sx', 'qx', 'taus', 'nu', 'w', 'klist', 'fcurve', 'cand', '-v7');
    O{r} = o;
    clear cTr cVa cTe m
end

T = struct2table([O{1:end-1}], 'AsArray', true);
writetable(T, fullfile(od, 'rounds.csv'));
J = O{end}; save(fullfile(od, 'joint.mat'), 'J'); writetable(struct2table(J, 'AsArray', true), fullfile(od, 'joint.csv'));
te_idx = te; save(fullfile(od, 'test_index.mat'), 'te_idx', 'yTe', '-v7');
end

function w = pick(s, names)
t = regexp(strtrim(split(s, ';')), '^(.*) \((\d+)\)$', 'tokens', 'once');
w = zeros(1, numel(names));
for i = 1:numel(t)
    j = find(names == t{i}{1});
    assert(isscalar(j), 'unknown feature %s', t{i}{1});
    w(j) = str2double(t{i}{2});
end
end
