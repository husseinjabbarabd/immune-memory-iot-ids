function run_select()
c = tl_cfg();
od = fullfile(c.res, 'components');
if ~exist(od, 'dir'); mkdir(od); end
tl_pool(c);
B = tl_base(c);
yTr = B.y(B.tr); nC = numel(B.names); one = ones(size(B.w)); out = {};

mi = tl_mi(B.cTr, yTr);
[~, o] = sort(mi, 'descend'); fm = sort(o(1:10));
inC = B.w > 0;
writetable(table(B.fn, mi', tiedrank(-mi'), inC', B.w', ismember((1:numel(mi))', fm), 'VariableNames', ...
    {'feature', 'mi_bits', 'mi_rank', 'clonalg', 'clonalg_weight', 'mi_top10'}), fullfile(od, 'select_features.csv'));

isB = B.lab == B.ben;
P = struct('f', B.fi, 'w', B.w, 'wexp', B.wexp, 'p', B.p, 'has', B.has);
sV = tl_score(B.cVa(:, B.fi), B.cells, isB, c.block);
[R, ex] = tl_eval(sV, B.sT, B, P, c);
out(end+1, :) = {'CLONALG features and weights', strjoin(B.fn(B.fi)', '; '), ex.auc, R};

P = struct('f', B.fi, 'w', one, 'wexp', one, 'p', zeros(numel(B.te), 1), 'has', false(numel(B.te), 1));
[R, ex] = tl_eval(sV, B.sT, B, P, c);
out(end+1, :) = {'CLONALG features, unit weights', strjoin(B.fn(B.fi)', '; '), ex.auc, R};

[cells, lab] = tl_airs2(B.cTr(:, fm), yTr, 1:nC, c);
isM = lab == B.ben;
sVm = tl_score(B.cVa(:, fm), cells, isM, c.block);
sTm = tl_score(B.cTe(:, fm), cells, isM, c.block);
P = struct('f', fm, 'w', one, 'wexp', one, 'p', zeros(numel(B.te), 1), 'has', false(numel(B.te), 1));
[R, ex] = tl_eval(sVm, sTm, B, P, c);
out(end+1, :) = {'Mutual information, unit weights', strjoin(B.fn(fm)', '; '), ex.auc, R};

L = cell(size(out, 1), 1);
for i = 1:size(out, 1)
    R = out{i, 4}; n = height(R);
    L{i} = [table(repmat(string(out{i, 1}), n, 1), repmat(string(out{i, 2}), n, 1), repmat(out{i, 3}, n, 1), ...
        'VariableNames', {'arm', 'features', 'auc'}), R];
end
writetable(vertcat(L{:}), fullfile(od, 'select_results.csv'));
end
