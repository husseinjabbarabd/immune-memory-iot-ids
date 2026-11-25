function run_memory()
c = tl_cfg(); c.km_iter = 100;
od = fullfile(c.res, 'components');
if ~exist(od, 'dir'); mkdir(od); end
tl_pool(c);
B = tl_base(c);
f = B.fi; Tr = B.cTr(:, f); yTr = B.y(B.tr); V = B.cVa(:, f); T = B.cTe(:, f);
P = struct('f', f, 'w', B.w, 'wexp', B.wexp, 'p', B.p, 'has', B.has);
nC = numel(B.names); out = {};

isB = B.lab == B.ben;
[sV, sT] = score2(V, T, B.cells, isB, c);
assert(isequal(sT, B.sT), 'AIRS2 test scores differ from the pipeline run');
[R, ex, taus] = tl_eval(sV, sT, B, P, c);
assert(isequal(taus, B.taus), 'AIRS2 thresholds differ from the pipeline run');
out(end+1, :) = {'AIRS2 cells', sum(isB), sum(~isB), ex.auc, R};

[U, ~, g] = unique(Tr, 'rows');
inB = accumarray(g, yTr == B.ben) > 0; inA = accumarray(g, yTr ~= B.ben) > 0;
s = tl_nn([V; T], U(inB, :), U(inA, :), c);
sV = s(1:size(V, 1)); sT = s(size(V, 1) + 1:end);
[R, ex] = tl_eval(sV, sT, B, P, c);
out(end+1, :) = {'Nearest neighbour (all training codes)', sum(inB), sum(inA), ex.auc, R};

cells = zeros(0, numel(f)); lab = zeros(0, 1); it = zeros(nC, 1);
for k = 1:nC
    nk = sum(B.lab == k);
    [Ck, it(k)] = tl_kmeans(Tr(yTr == k, :), nk, c.seed_airs + k, c);
    cells = [cells; Ck]; lab = [lab; k * ones(size(Ck, 1), 1)];
end
isK = lab == B.ben;
[sV, sT] = score2(V, T, cells, isK, c);
[R, ex] = tl_eval(sV, sT, B, P, c);
out(end+1, :) = {'k-means prototypes', sum(isK), sum(~isK), ex.auc, R};
writetable(table(B.names, it, accumarray(lab, 1, [nC 1]), 'VariableNames', {'class', 'iterations', 'centres'}), ...
    fullfile(od, 'memory_kmeans.csv'));

L = cell(size(out, 1), 1);
for i = 1:size(out, 1)
    R = out{i, 5}; n = height(R);
    L{i} = [table(repmat(string(out{i, 1}), n, 1), repmat(out{i, 2}, n, 1), repmat(out{i, 3}, n, 1), ...
        repmat(out{i, 4}, n, 1), 'VariableNames', {'arm', 'benign_vectors', 'attack_vectors', 'auc'}), R];
end
writetable(vertcat(L{:}), fullfile(od, 'memory_results.csv'));
end

function [sV, sT] = score2(V, T, cells, isB, c)
sV = tl_score(V, cells, isB, c.block); sT = tl_score(T, cells, isB, c.block);
end
