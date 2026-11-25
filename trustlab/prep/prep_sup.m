function prep_sup()
pth      = prep_paths();
SRC_S09  = fullfile(pth.out, 'S09',  'S09_features.mat');
SRC_S08B = fullfile(pth.out, 'S08b', 'S08b_split.mat');

OUT_DIR   = fullfile(pth.sup, 'SUP_S00');
DST_S09   = fullfile(OUT_DIR, 'S09_features.mat');
OUT_NAMES = fullfile(OUT_DIR, 'SUP_class_names.mat');
OUT_INFO  = fullfile(OUT_DIR, 'SUP_S00_input_info.mat');

N_EXPECTED = 4501678;
N_CLASSES  = 16;
REQ_VARS   = {'X', 'classId', 'fileId', 'rowInFile', 'featNames', 'idxFlow', 'idxPort', 'idxProto'};

if ~exist(OUT_DIR, 'dir'); mkdir(OUT_DIR); end
assert(exist(SRC_S09, 'file') == 2,  'Not found: %s', SRC_S09);
assert(exist(SRC_S08B, 'file') == 2, 'Not found: %s', SRC_S08B);

info  = whos('-file', SRC_S09);
names = {info.name};
missing = setdiff(REQ_VARS, names);
assert(isempty(missing), 'S09 is missing variables: %s', strjoin(missing, ', '));

hSrc = sha_file(SRC_S09);
[ok, msg] = copyfile(SRC_S09, DST_S09);
assert(ok, 'Copy failed: %s', msg);
hDst = sha_file(DST_S09);
assert(strcmp(hSrc, hDst), 'Copied file hash does not match source');
d1 = dir(SRC_S09); d2 = dir(DST_S09);
assert(d1.bytes == d2.bytes, 'File size mismatch');
s09Bytes = d2.bytes;

S = load(DST_S09, 'classId', 'featNames');
classId = double(S.classId(:));
assert(numel(classId) == N_EXPECTED, 'Row count %d ~= %d', numel(classId), N_EXPECTED);
assert(all(classId == round(classId)), 'classId not integer');
assert(min(classId) == 1 && max(classId) == N_CLASSES, 'classId range is not 1..%d', N_CLASSES);
assert(numel(unique(classId)) == N_CLASSES, 'Not all classes present');
nFeat = numel(S.featNames);
clear S classId;

C = load(SRC_S08B, 'classNames');
classNames = string(C.classNames(:));
clear C;
assert(numel(classNames) == N_CLASSES, 'classNames has %d entries', numel(classNames));
assert(sum(classNames == "Benign") == 1, 'Benign not found exactly once');
benignIdx = find(classNames == "Benign");
createdAt = datestr(now, 'yyyy-mm-dd HH:MM:SS');
srcNames  = SRC_S08B;
save(OUT_NAMES, 'classNames', 'benignIdx', 'srcNames', 'createdAt', '-v7');

srcS09  = SRC_S09;
copyS09 = DST_S09;
s09Hash = hDst;
save(OUT_INFO, 'srcS09', 'copyS09', 's09Hash', 's09Bytes', 'nFeat', 'createdAt', '-v7');
end
