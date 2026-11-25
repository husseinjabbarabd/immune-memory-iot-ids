function run_portfree()
c = tl_cfg();
drop = ["Src Port" "Dst Port"];
od = fullfile(c.res, 'portfree');
if ~exist(od, 'dir'); mkdir(od); end
tl_pool(c);

d = tl_data(c); X = d.X; y = d.y; sp = d.sp; fn = d.fn; names = d.names; ben = d.ben; clear d
nC = numel(names);
cand = find(~ismember(fn, drop))'; F = numel(cand);
assert(F == numel(fn) - 2, 'port features not found: %s', strjoin(fn', ', '));
tr = find(sp == 1); va = find(sp == 2); te = find(sp == 3);
yTr = y(tr); yVa = y(va); yTe = y(te);

m = tl_map(X(tr, cand));
cTr = tl_enc(X(tr, cand), m); cVa = tl_enc(X(va, cand), m);
[D, rI, qI] = tl_frame(cTr, yTr, 1:nC, ben, c);
K = parallel.pool.Constant(D); c2 = c; c2.klist(1) = F;
G = tl_clonalg(K, F, c2); clear K
wsel = G.wsel; fi = find(wsel > 0);
D2 = D; D2.tar = find(names == c.spec_cls); K2 = parallel.pool.Constant(D2);
[wexp, fe] = tl_spec(K2, wsel, c); clear K2 D D2

[cells, lab, AT] = tl_airs2(cTr(:, fi), yTr, 1:nC, c);
isB = lab == ben;
sV = tl_score(cVa(:, fi), cells, isB, c.block);
taus = tl_tau(sV, yVa ~= ben, c);

cTe = tl_enc(X(te, cand(fi)), m, fi);
[sT, qT] = tl_score(cTe, cells, isB, c.block);
trig = find(ismember(names, c.trig))';
iF = find(sT >= min(taus));
[p, st] = tl_p2(cTr(:, fi), yTr, cTe(iF, :), wsel(fi), wexp(fi), trig, nC, c);
pAll = ben * ones(numel(yTe), 1); pAll(iF) = p;
[RES, ex] = tl_metrics(sT, taus, pAll, yTe, ben, c);
writetable(RES, fullfile(od, 'closed_test_results.csv'));
for o = 1:3
    pc = array2table(ex.pc{o}, 'VariableNames', {'P', 'R', 'F1'}, 'RowNames', cellstr(names));
    writetable(pc, fullfile(od, sprintf('closed_attack_pc_op%d.csv', o)), 'WriteRowNames', true);
end
pr = pAll; pr(sT < taus(3)) = ben; F1 = zeros(nC, 1);
for k = 1:nC
    tp = sum(yTe == k & pr == k); P = tp / max(1, sum(pr == k)); R = tp / max(1, sum(yTe == k));
    if P + R > 0; F1(k) = 2 * P * R / (P + R); end
end
writetable(table(names, F1), fullfile(od, 'closed_e2e_f1_R95.csv'));
cmR = ex.cmR; auc = ex.auc; popA = ex.popA; popM = ex.popM; fa = ex.fa;
sel = fn(cand(fi)); wsel_fi = wsel(fi); wexp_fi = wexp(fi);
save(fullfile(od, 'test_rows.mat'), 'sT', 'qT', 'pAll', 'yTe', 'te', '-v7');
save(fullfile(od, 'closed_state.mat'), 'cmR', 'auc', 'popA', 'popM', 'fa', 'taus', 'sel', 'wsel_fi', 'wexp_fi', ...
    'AT', 'st', 'G', 'rI', 'qI', 'wsel', 'fi', 'fe', '-v7');

cTr = cTr(:, fi); vb = cVa(yVa == ben, fi); clear cVa
Z = [];
for x = setdiff(1:nC, ben)
    kp = yTr ~= x;
    [cx, lx, ATx] = tl_airs2(cTr(kp, :), yTr(kp), setdiff(1:nC, x), c);
    b = lx == ben;
    [sv, qv, dA, dB] = tl_score(vb, cx, b, c.block); nu = quantile(qv, 1 - c.alpha);
    tl_dist(fullfile(od, sprintf('zd%02d', x), 'score_val.mat'), dA, dB);
    [s, q, dA, dB] = tl_score(cTe, cx, b, c.block);
    tl_dist(fullfile(od, sprintf('zd%02d', x), 'score_test.mat'), dA, dB);
    isX = yTe == x; isBt = yTe == ben; isK = ~isX & ~isBt;
    r = struct('cls', names(x), 'cells', size(cx, 1), 'AT', ATx, 'nu', nu);
    for o = 1:3
        tau = taus(o); farV = mean(sv >= tau | qv >= nu);
        tM = tl_match(sv, farV); nM = tl_match(qv, farV);
        R = {s >= tau, s >= tau | q >= nu, s >= tM, q >= nM}; rl = 'aocq';
        r.(sprintf('tauM%d', o)) = tM; r.(sprintf('nuM%d', o)) = nM;
        for i = 1:4
            r.(sprintf('det_%s%d', rl(i), o)) = mean(R{i}(isX));
            r.(sprintf('far_%s%d', rl(i), o)) = mean(R{i}(isBt));
            r.(sprintf('kno_%s%d', rl(i), o)) = mean(R{i}(isK));
        end
    end
    Z = [Z, r];
    sx = single(s); qx = single(q);
    save(fullfile(od, sprintf('zd_%02d.mat', x)), 'sx', 'qx', 'nu', 'r', '-v7');
end
writetable(struct2table(Z), fullfile(od, 'zd_original_protocol.csv'));
clear cTr cTe X
run_nested('portfree', drop);
end
