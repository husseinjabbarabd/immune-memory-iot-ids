function prep_flow_split()
SEED          = 20261020;
TEST_FRAC     = 0.20;
VAL_FRAC      = 0.20;
N_EXPECTED    = 4501678;
N_CLASSES     = 16;
HASH_SUP_VEC  = '5fcbf3479a405460015dbdabaf7ef6a7dbd4c9eeee37852f355375c19169ab31';
HASH_S08_VEC  = 'f556628ee6e84f235c58704e56ee47c42ec4bd8b5ccf48f21c04177ab93b4267';

pth        = prep_paths();
IN_DIR     = fullfile(pth.fs, 'FS_S00');
S09_FILE   = fullfile(IN_DIR, 'S09_features.mat');
SUP_FILE   = fullfile(IN_DIR, 'SUP_S01_split.mat');
NAMES_FILE = fullfile(IN_DIR, 'SUP_class_names.mat');

OUT_DIR = fullfile(pth.fs, 'FS_S01');
OUT_MAT = fullfile(OUT_DIR, 'FS_S01_split.mat');
OUT_CSV = fullfile(OUT_DIR, 'FS_S01_class_counts.csv');
OUT_X   = fullfile(OUT_DIR, 'FS_S01_cross_tab.csv');
if ~exist(OUT_DIR, 'dir'); mkdir(OUT_DIR); end

S = load(S09_FILE, 'classId', 'fileId', 'classNames');
classId = double(S.classId(:)); fileId = S.fileId(:); namesS09 = string(S.classNames(:));
clear S;
E = load(SUP_FILE, 'split', 'hashSplitSup');
split2 = double(E.split(:));
assert(strcmp(E.hashSplitSup, HASH_SUP_VEC), 'Stored SUP_S01 split hash differs');
assert(strcmp(sha_vec(uint8(split2)), HASH_SUP_VEC), 'SUP_S01 split vector changed');
clear E;
Cn = load(NAMES_FILE, 'classNames', 'benignIdx');
classNames = string(Cn.classNames(:)); benignIdx = Cn.benignIdx; clear Cn;
N = numel(classId);

assert(N == N_EXPECTED, 'Row count %d ~= expected %d', N, N_EXPECTED);
assert(numel(fileId) == N && numel(split2) == N, 'Length mismatch');
assert(all(classId == round(classId)) && min(classId) == 1 && max(classId) == N_CLASSES, 'classId range');
assert(numel(unique(classId)) == N_CLASSES, 'Not all classes present');
assert(isequal(classNames, namesS09), 'Class names differ');
assert(classNames(benignIdx) == "Benign", 'benignIdx does not point to Benign');
assert(all(ismember(split2, [1 2 3])), 'SUP_S01 split has invalid codes');

split  = make_split(classId, split2, N_CLASSES, SEED, TEST_FRAC, VAL_FRAC);
splitB = make_split(classId, split2, N_CLASSES, SEED, TEST_FRAC, VAL_FRAC);
assert(isequal(split, splitB), 'Split is not deterministic for the same seed'); clear splitB;

classTotal = accumarray(classId, 1, [N_CLASSES 1]);
cnt = zeros(N_CLASSES, 3);
for s = 1:3; cnt(:, s) = accumarray(classId(split == s), 1, [N_CLASSES 1]); end
assert(all(split >= 1 & split <= 3), 'Unassigned rows exist');
assert(isequal(sum(cnt, 2), classTotal), 'Per-class counts do not sum to class totals');
assert(all(cnt(:) > 0), 'Some class has zero rows in some split');
rowsTr = find(split == 1); rowsVa = find(split == 2); rowsTe = find(split == 3);
assert(isempty(intersect(rowsTr, rowsVa)) && isempty(intersect(rowsTr, rowsTe)) && ...
    isempty(intersect(rowsVa, rowsTe)), 'Row overlap between splits');
assert(sum(split == 3 & split2 == 3) == 0, 'Test contains SUP_S01 Test rows');
assert(sum(split == 3 & split2 == 2) == 0, 'Test contains SUP_S01 Validation rows');

hashSplitFS = sha_vec(uint8(split));
assert(~strcmp(hashSplitFS, HASH_SUP_VEC) && ~strcmp(hashSplitFS, HASH_S08_VEC), 'Hash equals an earlier split');

splitNames = {'Train', 'Val', 'Test'};
X3 = zeros(3, 3);
for a = 1:3; for b = 1:3; X3(a, b) = sum(split == a & split2 == b); end; end

T = table((1:N_CLASSES)', classNames, classTotal, cnt(:, 1), cnt(:, 2), cnt(:, 3), ...
    cnt(:, 1) ./ classTotal, cnt(:, 2) ./ classTotal, cnt(:, 3) ./ classTotal, ...
    'VariableNames', {'classId', 'className', 'total', 'train', 'val', 'test', 'fracTrain', 'fracVal', 'fracTest'});
writetable(T, OUT_CSV);
TX = table(string(splitNames(:)), X3(:, 1), X3(:, 2), X3(:, 3), ...
    'VariableNames', {'split', 'sup_train', 'sup_val', 'sup_test'});
writetable(TX, OUT_X);

split = uint8(split); rowId = uint32((1:N)'); classIdU8 = uint8(classId); counts = cnt; crossTab = X3;
createdAt = datestr(now, 'yyyy-mm-dd HH:MM:SS');
save(OUT_MAT, 'split', 'rowId', 'classIdU8', 'classNames', 'benignIdx', 'counts', 'classTotal', 'crossTab', ...
    'SEED', 'TEST_FRAC', 'VAL_FRAC', 'hashSplitFS', 'createdAt', '-v7');
end

function split = make_split(classId, split2, nClasses, seed, testFrac, valFrac)
rs = RandStream('twister', 'Seed', seed);
N = numel(classId); split = zeros(N, 1, 'uint8');
for c = 1:nClasses
    idx  = find(classId == c);
    n    = numel(idx);
    nTe  = round(testFrac * n);
    pool = idx(split2(idx) == 1);
    assert(numel(pool) >= nTe, 'Class %d: not enough SUP_S01 Train rows for the Test', c);
    te   = pool(randperm(rs, numel(pool), nTe));
    rest = setdiff(idx, te);
    nVa  = round(valFrac * (n - nTe));
    va   = rest(randperm(rs, numel(rest), nVa));
    split(idx) = 1; split(te) = 3; split(va) = 2;
end
end
