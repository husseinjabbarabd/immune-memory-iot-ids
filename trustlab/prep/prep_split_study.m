function prep_split_study()
pth    = prep_paths();
S08DIR = fullfile(pth.out, 'S08');
OUT    = fullfile(pth.out, 'S08b');
if ~exist(OUT, 'dir'), mkdir(OUT); end

SEED          = 20260402;
P_TEST_ATTACK = 0.30;
N_TEST_BENIGN = 436332;
P_VAL         = 0.20;
BENIGN        = 2;

S = load(fullfile(S08DIR, 'S08_split.mat'), 'classId', 'fileId', 'rowInFile', 'classNames', ...
         'fileNames');
classId    = S.classId;
fileId     = S.fileId;
rowInFile  = S.rowInFile;
classNames = S.classNames;
fileNames  = S.fileNames;
N   = numel(classId);
nCls = numel(classNames);
if ~strcmp(char(classNames(BENIGN)), 'Benign')
    error('Class index %d is not Benign.', BENIGN);
end
total = accumarray(double(classId), 1, [nCls 1])';

rng(SEED, 'twister');
split = zeros(N, 1, 'uint8');
for c = 1:nCls
    idx = find(classId == c);
    n   = numel(idx);
    p   = idx(randperm(n));
    if c == BENIGN
        nTest = N_TEST_BENIGN;
    else
        nTest = round(P_TEST_ATTACK * n);
    end
    nDev = n - nTest;
    nVal = round(P_VAL * nDev);
    split(p(1:nTest))            = 3;
    split(p(nTest+1:nTest+nVal)) = 2;
    split(p(nTest+nVal+1:end))   = 1;
end

if any(split == 0), error('Some rows were not assigned.'); end
T = accumarray([double(classId) double(split)], 1, [nCls 3]);
if any(sum(T, 2)' ~= total) || sum(T(:)) ~= N, error('Counts do not add up.'); end

hashSplit = sha_vec(split);

splitNames = ["Train", "Validation", "Test"];
save(fullfile(OUT, 'S08b_split.mat'), 'classId', 'fileId', 'rowInFile', 'split', 'classNames', ...
     'fileNames', 'splitNames', 'SEED', 'P_TEST_ATTACK', 'N_TEST_BENIGN', 'P_VAL', 'hashSplit', '-v7.3');

tbl = table(classNames(:), total(:), T(:,1), T(:,2), T(:,3), 100*T(:,3)./total(:), ...
    'VariableNames', {'class','total','train','validation','test','test_pct'});
nTestB = T(BENIGN, 3);
nTestA = sum(T(:,3)) - nTestB;
tot = table("ALL", sum(total), sum(T(:,1)), sum(T(:,2)), sum(T(:,3)), 100*sum(T(:,3))/N, ...
    'VariableNames', tbl.Properties.VariableNames);
writetable([tbl; tot], fullfile(OUT, 'S08b_split_counts.csv'));

fid = fopen(fullfile(OUT, 'S08b_summary.txt'), 'w');
pr = @(varargin) fprintf(fid, varargin{:});
pr('S08b - Split with the test composition of the original study\n');
pr('=============================================================\n');
pr('Rows assigned: %d | seed %d (twister)\n', N, SEED);
pr('Train / Val / Test rows : %d / %d / %d\n', sum(T(:,1)), sum(T(:,2)), sum(T(:,3)));
pr('Test Benign / Attack    : %d / %d  (attack share of test = %.1f%%)\n', nTestB, nTestA, 100*nTestA/sum(T(:,3)));
pr('SHA-256 split vector    : %s\n\n', hashSplit);
pr('Per-class counts (Train / Validation / Test):\n');
for c = 1:nCls
    pr('  %-15s total %8d | train %8d | val %7d | test %7d (%.2f%%)\n', classNames(c), total(c), ...
        T(c,1), T(c,2), T(c,3), 100*T(c,3)/total(c));
end
pr('\nChecks: every row assigned once; counts equal the class totals.\n');
fclose(fid);

end
