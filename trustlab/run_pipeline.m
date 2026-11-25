function run_pipeline(published)
if nargin < 1; published = false; end
c = tl_cfg();
od = fullfile(c.res, 'pipeline');
if ~exist(od, 'dir'); mkdir(od); end
ck = {};
tl_pool(c);

S = load(c.s09, 'classId'); y = double(S.classId(:));
E = load(c.sup, 'split'); sp = tl_split(y, double(E.split(:)), c); clear S E
N = load(c.names, 'classNames', 'benignIdx'); names = string(N.classNames(:)); ben = N.benignIdx;
ck(end+1, :) = {'split hash', tl_sha(sp), 'c91ec90b933f612aa3112f3e9082e405ff01d25a71b31b8ea2efa2553923f85f'};

S = load(c.s09, 'X', 'featNames'); X = S.X; fn = string(S.featNames(:)); clear S
tr = find(sp == 1); va = find(sp == 2); te = find(sp == 3);
yTr = y(tr); yVa = y(va); yTe = y(te);
m = tl_map(X(tr, :));
cTr = tl_enc(X(tr, :), m); cVa = tl_enc(X(va, :), m);
ck(end+1, :) = {'train codes', tl_sha(typecast(cTr(:), 'uint8')), 'c82cdfb8a944cefd59d785ce633f3563476cbcd530653f12a8067d45b89a2d38'};
ck(end+1, :) = {'val codes', tl_sha(typecast(cVa(:), 'uint8')), '062148bac8a67c0958fd49e0ed1e32b38b7d4600e2db9043468f1c09865b6921'};

ref = fullfile(c.ref, 'FS_S03');
[D, rI, qI] = tl_frame(cTr, yTr, 1:16, ben, c);
Sel = load(fullfile(ref, 'FS_S03_selection.mat'), 'rIdx', 'qIdx', 'wChosen');
ck(end+1, :) = {'frame rows', tl_sha(typecast(uint32([rI; qI]), 'uint8')), tl_sha(typecast(uint32([Sel.rIdx(:); Sel.qIdx(:)]), 'uint8'))};
if published
    G = []; fe = [];
    wsel = double(Sel.wChosen(:)');
    W = load(fullfile(c.ref, 'FS_S04', 'FS_S04_weights_exp.mat'), 'weightsExpFS'); wexp = double(W.weightsExpFS(:)');
else
    K = parallel.pool.Constant(D);
    G = tl_clonalg(K, 74, c);
    for ik = 1:numel(G.klist)
        R = load(fullfile(ref, sprintf('FS_S03_k%02d.mat', G.klist(ik))), 'wBest', 'fBest', 'hist');
        ck(end+1, :) = {sprintf('k%02d weights+fitness+hist', G.klist(ik)), ...
            mat2str([G.w(ik, :), G.f(ik), G.hist{ik}(:)'], 12), mat2str([R.wBest, R.fBest, R.hist(:)'], 12)};
    end
    wsel = G.wsel;
    D2 = D; D2.tar = find(names == c.spec_cls);
    K2 = parallel.pool.Constant(D2);
    [wexp, fe] = tl_spec(K2, wsel, c);
    clear K K2
end
ck(end+1, :) = {'selected weights hash', tl_sha(uint8(wsel)), 'd6102fbde2b4ed284492e35f2a529a93a1913cf0005d37c2e861754236e15c80'};
ck(end+1, :) = {'specialist hash', tl_sha(uint8(wexp)), 'd2d70031115f865a580e400b3141e56eab666eccbdeaab4486ed3738cdc6a96b'};
fi = find(wsel > 0);
sel = fn(fi);
save(fullfile(od, 'clonalg.mat'), 'G', 'sel', 'wexp', 'fe', '-v7');

[cells, lab, AT] = tl_airs2(cTr(:, fi), yTr, 1:16, c);
ck(end+1, :) = {'AIRS2 cells hash', tl_sha([typecast(uint16(cells(:)), 'uint8'); uint8(lab(:))]), '9bc5ad0c5ed61945dc4282c36dfca7c789d3e0971f4af8c230145eb86099ccd5'};
ck(end+1, :) = {'AT', sprintf('%.6f', AT), '0.375085'};

isB = lab == ben;
sV = tl_score(cVa(:, fi), cells, isB, c.block);
taus = tl_tau(sV, yVa ~= ben, c);
V = load(fullfile(c.ref, 'FS_S06', 'FS_S06_val_state.mat'), 'tauS', 'tauF2', 'tauR');
ck(end+1, :) = {'thresholds (exact doubles)', sprintf('%.17g ', taus), sprintf('%.17g ', [V.tauS V.tauF2 V.tauR])};

cTe = tl_enc(X(te, fi), m, fi); clear X
[sT, qT, dA, dB] = tl_score(cTe, cells, isB, c.block);
RS = readtable(fullfile(c.ref, 'FS_S07', 'FS_S07_TEST_results.csv'));
fw = arrayfun(@(o) sum(sT >= taus(o)), 1:3);
ck(end+1, :) = {'test forwarded counts', mat2str(fw), mat2str(RS.nFwd')};
ties = sum(dA == 0 & dB == 0);

trig = find(ismember(names, c.trig))';
iF = find(sT >= min(taus));
[p, st] = tl_p2(cTr(:, fi), yTr, cTe(iF, :), wsel(fi), wexp(fi), trig, 16, c);
pAll = ben * ones(numel(yTe), 1); pAll(iF) = p;
ST = load(fullfile(c.ref, 'FS_S07', 'FS_S07_TEST_state.mat'), 'predFsave', 'iFSsave', 'auc', 'pA', 'pM', 'faAll', 'cmR');
ck(end+1, :) = {'exact memory hash', st.memhash, 'fd7cf7b79f2f3d8339e1bda2fe118f100c9ac00716f096ed08e2f0d82880e66c'};
ck(end+1, :) = {'phase-2 predictions', tl_sha([typecast(uint32(iF), 'uint8'); typecast(int8(p), 'uint8')]), ...
    tl_sha([typecast(ST.iFSsave(:), 'uint8'); typecast(ST.predFsave(:), 'uint8')])};
[RES, ex] = tl_metrics(sT, taus, pAll, yTe, ben, c);
num = RES{:, 2:end}; numR = RS{:, {'tau', 'nFwd', 'P1_Acc', 'P1_MF1', 'P1_Rec', 'P1_Spec', 'P1_F2', 'P1_Acc_pc', ...
    'fwd_Acc', 'fwd_MF1', 'fwd_BenignF1', 'fwd_Acc_pc', 'fwd_MF1_pc', 'att_n', 'att_Acc', 'att_MF1', 'e2e_Acc', 'e2e_MF1', 'e2e_Acc_pc'}};
dmax = max(abs(num(:) - numR(:)));
ck(end+1, :) = {'test results table (|diff| < 1e-12)', sprintf('%d', dmax < 1e-12), '1'};
ck(end+1, :) = {'AUC / POP-MATCH / false alarms', sprintf('%.15g ', [ex.auc ex.popA ex.popM ex.fa]), sprintf('%.15g ', [ST.auc ST.pA ST.pM ST.faAll])};
ck(end+1, :) = {'R95 confusion matrix', tl_sha(typecast(ex.cmR(:), 'uint8')), tl_sha(typecast(double(ST.cmR(:)), 'uint8'))};
writetable(RES, fullfile(od, 'test_results.csv'));
save(fullfile(od, 'test_rows.mat'), 'sT', 'qT', 'pAll', 'yTe', 'te', 'taus', 'ties', 'st', '-v7');
[mem, ~, g] = unique(cTr(:, fi), 'rows');
[~, mlab] = max(accumarray([g, yTr], 1, [size(mem, 1), 16]), [], 2);
ck(end+1, :) = {'exported memory hash', tl_sha([typecast(mem(:), 'uint8'); uint8(mlab(:))]), st.memhash};
map = struct('u', {m.u(fi)}, 'less', {m.less(fi)}, 'cnt', {m.cnt(fi)}, 'n', m.n);
save(fullfile(od, 'model.mat'), 'map', 'fi', 'sel', 'wsel', 'wexp', 'cells', 'lab', 'AT', 'taus', 'mem', 'mlab', '-v7');

vb = cVa(yVa == ben, fi);
Z = []; att = setdiff(1:16, ben);
for x = att
    kp = yTr ~= x;
    [cx, lx, ATx] = tl_airs2(cTr(kp, fi), yTr(kp), setdiff(1:16, x), c);
    b = lx == ben;
    [sv, qv, dA, dB] = tl_score(vb, cx, b, c.block); nu = quantile(qv, 1 - c.alpha);
    tl_dist(fullfile(od, sprintf('zd%02d', x), 'score_val.mat'), dA, dB);
    [s, q, dA, dB] = tl_score(cTe, cx, b, c.block);
    tl_dist(fullfile(od, sprintf('zd%02d', x), 'score_test.mat'), dA, dB);
    isX = yTe == x; isBt = yTe == ben;
    r = struct('cls', names(x), 'cells', size(cx, 1), 'AT', ATx, 'nu', nu);
    for o = 1:3
        r.(sprintf('det%d', o)) = mean(s(isX) >= taus(o)); r.(sprintf('far%d', o)) = mean(s(isBt) >= taus(o));
        r.(sprintf('detOR%d', o)) = mean(s(isX) >= taus(o) | q(isX) >= nu);
        r.(sprintf('farOR%d', o)) = mean(s(isBt) >= taus(o) | q(isBt) >= nu);
        tm = tl_match(sv, mean(sv >= taus(o) | qv >= nu));
        r.(sprintf('tauM%d', o)) = tm; r.(sprintf('detM%d', o)) = mean(s(isX) >= tm); r.(sprintf('farM%d', o)) = mean(s(isBt) >= tm);
    end
    rk = tiedrank(s([find(isBt); find(isX)])); nP = sum(isX); nN = sum(isBt);
    r.auc = (sum(rk(nN+1:end)) - nP * (nP + 1) / 2) / (nP * nN);
    sx = single(s); qx = single(q);
    save(fullfile(od, sprintf('zd_%02d.mat', x)), 'sx', 'qx', 'nu', 'r', '-v7');
    Z = [Z, r];
    R1 = load(fullfile(c.refzd, 'ZD_S01', sprintf('ZD_S01_round_%02d_%s.mat', x, names(x))), 'r'); R1 = R1.r;
    R2 = load(fullfile(c.refzd, 'ZD_S02', sprintf('ZD_S02_round_%02d_%s.mat', x, names(x))), 'r'); R2 = R2.r;
    R3 = load(fullfile(c.refzd, 'ZD_S03', sprintf('ZD_S03_round_%02d_%s.mat', x, names(x))), 'r'); R3 = R3.r;
    a = [r.cells r.det1 r.det2 r.det3 r.far1 r.far3 r.auc r.nu r.detOR1 r.detOR3 r.farOR1 r.farOR3 r.tauM1 r.tauM3 r.detM1 r.detM3 r.farM1 r.farM3];
    bR = [R1.cells R1.detTest_1 R1.detTest_2 R1.detTest_3 R1.far_1 R1.far_3 R1.aucTest R2.nu R2.detOR_1 R2.detOR_3 R2.farOR_1 R2.farOR_3 ...
        R3.tauM_1 R3.tauM_3 R3.detM_1 R3.detM_3 R3.farM_1 R3.farM_3];
    ck(end+1, :) = {sprintf('ZD %s (|diff| < 1e-12)', names(x)), sprintf('%d', max(abs(a - bR)) < 1e-12), '1'};
end
writetable(struct2table(Z), fullfile(od, 'zd_original_protocol.csv'));

CK = cell2table(ck, 'VariableNames', {'item', 'run', 'reference'});
CK.match = strcmp(CK.run, CK.reference);
writetable(CK, fullfile(od, 'checks.csv'));
fprintf('%d of %d checks match the reference outputs\n', sum(CK.match), height(CK));
end
