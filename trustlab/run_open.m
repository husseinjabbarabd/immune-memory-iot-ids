function run_open()
c = tl_cfg(); c.km_iter = 100;
src = fullfile(c.res, 'nested'); od = fullfile(c.res, 'components');
if ~exist(od, 'dir'); mkdir(od); end
tl_pool(c);
d = tl_data(c); X = d.X; y = d.y; sp = d.sp; names = d.names; ben = d.ben; clear d
rounds = [num2cell(setdiff(1:numel(names), ben)), {find(ismember(names, c.twins))'}];
yTe = y(sp == 3);
rows = []; joint = [];
for r = 1:numel(rounds)
    x = rounds{r}; rn = strjoin(names(x), '+');
    sd = fullfile(src, sprintf('r%02d', r));
    L = load(fullfile(src, sprintf('round_%02d.mat', r)), 'o', 'w', 'cand', 'taus', 'nu');
    cols = L.cand(L.w > 0);
    tr = find(sp == 1 & ~ismember(y, x)); va = find(sp == 2 & ~ismember(y, x));
    m = tl_map(X(tr, cols));
    Tr = tl_enc(X(tr, cols), m); Va = tl_enc(X(va, cols), m); Te = tl_enc(X(sp == 3, cols), m);
    A = load(fullfile(sd, 'airs2.mat'), 'cells', 'lab'); lab = A.lab;
    check(Va, A.cells, lab == ben, fullfile(sd, 'score_val.mat'), r);
    yTr = y(tr); Vb = Va(y(va) == ben, :); clear Va
    taus = L.taus; nu = L.nu; far = [L.o.vfar_o1 L.o.vfar_o2 L.o.vfar_o3]; ref = L.o;
    assert(~any(ismember(yTr, x)), 'round %d: held-out family in the training rows', r);
    [s, q] = eq6(fullfile(sd, 'score_test.mat'), size(Te, 2));
    O = tl_osr(Tr, yTr, Vb, Te, ben, sum(lab ~= ben), far, c.seed_airs + [800 900] + r, c);
    isX = ismember(yTe, x); isB = yTe == ben; isK = ~isX & ~isB;
    z = struct('round', r, 'family', rn, 'nX', sum(isX), 'detectors', O.detectors, 'max_detectors', O.max_detectors);
    j = struct('round', {}, 'class', {}, 'rule', {}, 'det', {});
    F = [arrayfun(@(o) {sprintf('combined%d', o), s >= taus(o) | q >= nu, far(o)}, 1:3, 'UniformOutput', false)'; ...
        num2cell(O.flag, 2)];
    for k = 1:numel(F)
        [nm, f, vf] = F{k}{:};
        z.(['det_' nm]) = mean(f(isX)); z.(['far_' nm]) = mean(f(isB)); z.(['kno_' nm]) = mean(f(isK)); z.(['vfar_' nm]) = vf;
        if numel(x) > 1
            for i = x; j(end+1) = struct('round', r, 'class', names(i), 'rule', nm, 'det', mean(f(yTe == i))); end
        end
    end
    for o = 1:3
        assert(z.(sprintf('det_combined%d', o)) == ref.(sprintf('det_o%d', o)), 'round %d: combined rule differs', r);
    end
    rows = [rows; z]; joint = [joint; j(:)];
end
writetable(struct2table(rows), fullfile(od, 'open_nested.csv'));
writetable(struct2table(joint), fullfile(od, 'open_nested_joint.csv'));
end

function check(Va, cells, isB, f, r)
S = load(f, 'dA', 'dB');
i = sort(randperm(RandStream('twister', 'Seed', r), size(Va, 1), min(20000, size(Va, 1))));
[~, ~, dA, dB] = tl_score(Va(i, :), cells, isB, 1e7);
assert(isequal(dA, S.dA(i)) && isequal(dB, S.dB(i)), 'round %d: coding does not reproduce the saved scores', r);
end

function [s, q] = eq6(f, d)
S = load(f, 'dA', 'dB'); sA = sqrt(S.dA); sB = sqrt(S.dB);
s = sB ./ (sA + sB); s(sA + sB == 0) = 0.5; q = sB / (1023 * sqrt(d));
end
