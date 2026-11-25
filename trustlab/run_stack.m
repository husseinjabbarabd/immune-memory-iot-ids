function run_stack()
c = tl_cfg();
od = fullfile(c.res, 'components'); wd = fullfile(od, 'stack');
if ~exist(wd, 'dir'); mkdir(wd); end
B = tl_base(c); f = B.fi;
Xtr = B.cTr(:, f); ytr = B.y(B.tr); Xva = B.cVa(:, f); yva = B.y(B.va); Xte = B.cTe(:, f); yte = B.y(B.te);
ben = B.ben; feat = cellstr(B.fn(f)); clear B
xf = fullfile(wd, 'data.mat');
save(xf, 'Xtr', 'ytr', 'Xva', 'yva', 'Xte', 'yte', 'ben', 'feat', '-v7');
py = getenv('PYTHON'); if isempty(py); py = 'python'; end
cmd = sprintf('"%s" "%s" "%s" "%s" %d', py, fullfile(fileparts(mfilename('fullpath')), 'py', 'stack.py'), xf, wd, c.workers);
assert(system(cmd) == 0, 'stack.py failed');
P = load(fullfile(wd, 'scores.mat'), 'pV', 'pT', 'lab');
taus = tl_tau(P.pV(:), yva ~= ben, c);
[R, ex] = tl_metrics(P.pT(:), taus, double(P.lab(:)), yte, ben, c);
R = [table(repmat(ex.auc, height(R), 1), 'VariableNames', {'auc'}), R];
writetable(R, fullfile(od, 'stack_results.csv'));
end
