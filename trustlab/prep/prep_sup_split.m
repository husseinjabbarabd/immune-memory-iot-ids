function prep_sup_split()
SEED         = 20261008;
TEST_FRAC    = 0.20;
VAL_FRAC     = 0.20;
N_EXPECTED   = 4501678;
N_CLASSES    = 16;
HASH_S08_VEC = 'f556628ee6e84f235c58704e56ee47c42ec4bd8b5ccf48f21c04177ab93b4267';

pth        = prep_paths();
IN_DIR     = fullfile(pth.sup, 'SUP_S00');
S09_FILE   = fullfile(IN_DIR, 'S09_features.mat');
NAMES_FILE = fullfile(IN_DIR, 'SUP_class_names.mat');
INFO_FILE  = fullfile(IN_DIR, 'SUP_S00_input_info.mat');

OUT_DIR = fullfile(pth.sup, 'SUP_S01');
OUT_MAT = fullfile(OUT_DIR, 'SUP_S01_split.mat');
OUT_CSV = fullfile(OUT_DIR, 'SUP_S01_class_counts.csv');
if ~exist(OUT_DIR, 'dir'); mkdir(OUT_DIR); end

I = load(INFO_FILE, 's09Hash');
assert(strcmp(sha_file(S09_FILE), I.s09Hash), 'S09 copy hash changed since prep_sup');
clear I;

S = load(S09_FILE, 'classId', 'fileId', 'classNames');
classId = double(S.classId(:));
fileId  = S.fileId(:);
namesS09 = string(S.classNames(:));
clear S;

Cn = load(NAMES_FILE, 'classNames', 'benignIdx');
classNames = string(Cn.classNames(:));
benignIdx  = Cn.benignIdx;
clear Cn;

N = numel(classId);
assert(N == N_EXPECTED, 'Row count %d ~= expected %d', N, N_EXPECTED);
assert(numel(fileId) == N, 'fileId length mismatch');
assert(all(isfinite(classId)) && all(classId == round(classId)), 'classId not integer');
assert(min(classId) == 1 && max(classId) == N_CLASSES, 'classId range is not 1..%d', N_CLASSES);
assert(numel(unique(classId)) == N_CLASSES, 'Not all %d classes present', N_CLASSES);
assert(numel(classNames) == N_CLASSES, 'classNames has %d entries', numel(classNames));
assert(isequal(classNames, namesS09), 'Class names in S09 and SUP_class_names differ');
assert(classNames(benignIdx) == "Benign", 'benignIdx does not point to Benign');

split  = make_split(classId, N_CLASSES, SEED, TEST_FRAC, VAL_FRAC);
split2 = make_split(classId, N_CLASSES, SEED, TEST_FRAC, VAL_FRAC);
assert(isequal(split, split2), 'Split is not deterministic for the same seed');
clear split2;

classTotal = accumarray(classId, 1, [N_CLASSES 1]);
cnt = zeros(N_CLASSES, 3);
for s = 1:3
    cnt(:, s) = accumarray(classId(split == s), 1, [N_CLASSES 1]);
end
assert(all(split >= 1 & split <= 3), 'Unassigned rows exist');
assert(sum(cnt(:)) == N, 'Counts do not sum to N');
assert(isequal(sum(cnt, 2), classTotal), 'Per-class counts do not sum to class totals');
assert(all(cnt(:) > 0), 'Some class has zero rows in some split');

rowsTr = find(split == 1); rowsVa = find(split == 2); rowsTe = find(split == 3);
assert(isempty(intersect(rowsTr, rowsVa)) && isempty(intersect(rowsTr, rowsTe)) && ...
    isempty(intersect(rowsVa, rowsTe)), 'Row overlap between splits');
assert(numel(rowsTr) + numel(rowsVa) + numel(rowsTe) == N, 'Split sizes do not sum to N');

hashSplitSup = sha_vec(split);
assert(~strcmp(hashSplitSup, HASH_S08_VEC), 'Hash equals the S08 split hash');

T = table((1:N_CLASSES)', classNames, classTotal, cnt(:, 1), cnt(:, 2), cnt(:, 3), ...
    cnt(:, 1) ./ classTotal, cnt(:, 2) ./ classTotal, cnt(:, 3) ./ classTotal, ...
    'VariableNames', {'classId', 'className', 'total', 'train', 'val', 'test', ...
    'fracTrain', 'fracVal', 'fracTest'});
writetable(T, OUT_CSV);

split     = uint8(split);
rowId     = uint32((1:N)');
classIdU8 = uint8(classId);
counts    = cnt;
createdAt = datestr(now, 'yyyy-mm-dd HH:MM:SS');
srcS09    = S09_FILE;
save(OUT_MAT, 'split', 'rowId', 'classIdU8', 'classNames', 'benignIdx', 'counts', 'classTotal', ...
    'SEED', 'TEST_FRAC', 'VAL_FRAC', 'hashSplitSup', 'createdAt', 'srcS09', '-v7');
end

function split = make_split(classId, nClasses, seed, testFrac, valFrac)
rng(seed, 'twister');
N = numel(classId);
split = zeros(N, 1, 'uint8');
for c = 1:nClasses
    idx = find(classId == c);
    n   = numel(idx);
    p   = idx(randperm(n));
    nTe = round(testFrac * n);
    nVa = round(valFrac * (n - nTe));
    split(p(1:nTe))            = 3;
    split(p(nTe+1:nTe+nVa))    = 2;
    split(p(nTe+nVa+1:end))    = 1;
end
end
