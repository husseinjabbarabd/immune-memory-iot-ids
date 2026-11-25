function nested_boot()
c = tl_cfg(); wd = fullfile(c.res, 'nested');
T = load(fullfile(wd, 'test_index.mat'), 'yTe'); yTe = T.yTe(:);
N = load(c.names, 'classNames', 'benignIdx'); names = string(N.classNames(:)); ben = N.benignIdx;
att = setdiff(1:numel(names), ben);
iB = find(yTe == ben); nB = numel(iB);

U = struct('round', {}, 'family', {}, 'cls', {});
for r = 1:15; U(end + 1) = struct('round', r, 'family', names(att(r)), 'cls', att(r)); end
for z = c.twins; U(end + 1) = struct('round', 16, 'family', z + "(joint)", 'cls', find(names == z)); end
nu_ = numel(U); B = 2000; ops = ["OP-SPEC" "OP-F2" "OP-R95"];
DX = cell(nu_, 3, 2); DB = zeros(nB, 16 * 3 * 2); pt = zeros(nu_, 3, 2, 2);
for r = 1:16
    L = load(fullfile(wd, sprintf('round_%02d.mat', r)), 'o', 'taus', 'nu', 'w');
    S = load(fullfile(wd, sprintf('r%02d', r), 'score_test.mat'), 'dA', 'dB');
    assert(numel(S.dA) == numel(yTe), 'round %d: score size', r);
    k = nnz(L.w > 0); sB = sqrt(S.dB); sA = sqrt(S.dA);
    s = sB ./ (sA + sB); s(sA + sB == 0) = 0.5; q = sB / (1023 * sqrt(k));
    for j = 1:3
        fu = s >= L.taus(j) | q >= L.nu;
        R = {s >= L.o.(sprintf('tauM%d', j)), q >= L.o.(sprintf('nuM%d', j))};
        for h = 1:2
            DB(:, ((r - 1) * 3 + j - 1) * 2 + h) = double(fu(iB)) - double(R{h}(iB));
            assert(abs(mean(fu(iB)) - L.o.(sprintf('far_o%d', j))) < 1e-12, 'round %d: far not reproduced', r);
        end
        for u = find([U.round] == r)
            ix = yTe == U(u).cls;
            if r < 16; assert(mean(fu(ix)) == L.o.(sprintf('det_o%d', j)), 'unit %d det not reproduced', u);
            else; assert(mean(fu(ix)) == L.o.(sprintf('det_%s_%d', names(U(u).cls), j)), 'unit %d det not reproduced', u); end
            for h = 1:2
                DX{u, j, h} = double(fu(ix)) - double(R{h}(ix));
                pt(u, j, h, 1) = mean(DX{u, j, h});
                pt(u, j, h, 2) = mean(DB(:, ((r - 1) * 3 + j - 1) * 2 + h));
            end
        end
    end
end
DB = sparse(DB);
bt = zeros(B, nu_, 3, 2, 2);
rng(7, 'twister');
for b = 1:B
    wB = accumarray(randi(nB, nB, 1), 1, [nB 1]);
    fb = full(wB' * DB) / nB;
    for u = 1:nu_
        nx = numel(DX{u, 1, 1}); wX = accumarray(randi(nx, nx, 1), 1, [nx 1]);
        for j = 1:3
            for h = 1:2
                bt(b, u, j, h, 1) = wX' * DX{u, j, h} / nx;
                bt(b, u, j, h, 2) = fb(((U(u).round - 1) * 3 + j - 1) * 2 + h);
            end
        end
    end
end
fam = [U.family]; s13 = find(~ismember(fam, c.twins) & [U.round] < 16); s15 = 1:15;
s15j = [s13, find([U.round] == 16)];
sets = {'i_13', s13; 'ii_15', s15; 'iii_15_joint', s15j};
cn = ["full-control" "full-arm"]; mn = ["det" "far"];
fo = fopen(fullfile(wd, 'boot.txt'), 'w');
fprintf(fo, 'bootstrap over flows, B = %d, seed 7 (twister), one shared Benign resample per replicate\n', B);
fprintf(fo, 'values in percentage points: point [2.5%%, 97.5%%]\n\n');
rowsT = {};
for j = 1:3
    for h = 1:2
        for m = 1:2
            for u = 1:nu_
                ci = prctile(bt(:, u, j, h, m), [2.5 97.5]) * 100;
                rowsT(end + 1, :) = {ops(j), cn(h), mn(m), fam(u), numel(DX{u, 1, 1}), 100 * pt(u, j, h, m), ci(1), ci(2)};
            end
            for g = 1:size(sets, 1)
                v = mean(bt(:, sets{g, 2}, j, h, m), 2); ci = prctile(v, [2.5 97.5]) * 100;
                rowsT(end + 1, :) = {ops(j), cn(h), mn(m), "mean " + sets{g, 1}, NaN, 100 * mean(pt(sets{g, 2}, j, h, m)), ci(1), ci(2)};
                fprintf(fo, '%s %s %s mean %s: %.2f [%.2f, %.2f]\n', ops(j), cn(h), mn(m), sets{g, 1}, ...
                    100 * mean(pt(sets{g, 2}, j, h, m)), ci(1), ci(2));
            end
        end
    end
end
fclose(fo);
Tb = cell2table(rowsT, 'VariableNames', {'op', 'contrast', 'measure', 'family', 'n_test', 'point_pp', 'lo_pp', 'hi_pp'});
writetable(Tb, fullfile(wd, 'boot.csv'));
end
