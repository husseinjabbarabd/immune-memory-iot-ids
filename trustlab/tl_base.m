function B = tl_base(c)
d = tl_data(c);
B.y = d.y; B.fn = d.fn; B.names = d.names; B.ben = d.ben;
B.tr = find(d.sp == 1); B.va = find(d.sp == 2); B.te = find(d.sp == 3);
m = tl_map(d.X(B.tr, :));
B.cTr = tl_enc(d.X(B.tr, :), m); B.cVa = tl_enc(d.X(B.va, :), m); B.cTe = tl_enc(d.X(B.te, :), m);
clear d m
assert(strcmp(tl_sha(typecast(B.cTr(:), 'uint8')), 'c82cdfb8a944cefd59d785ce633f3563476cbcd530653f12a8067d45b89a2d38'));

S = load(fullfile(c.ref, 'FS_S03', 'FS_S03_selection.mat'), 'wChosen'); B.w = double(S.wChosen(:)');
S = load(fullfile(c.ref, 'FS_S04', 'FS_S04_weights_exp.mat'), 'weightsExpFS'); B.wexp = double(S.weightsExpFS(:)');
assert(strcmp(tl_sha(uint8(B.w)), 'd6102fbde2b4ed284492e35f2a529a93a1913cf0005d37c2e861754236e15c80'));
B.fi = find(B.w > 0);

pd = fullfile(c.res, 'pipeline');
A = load(fullfile(pd, 'model.mat'), 'cells', 'lab'); B.cells = A.cells; B.lab = A.lab;
assert(strcmp(tl_sha([typecast(uint16(B.cells(:)), 'uint8'); uint8(B.lab(:))]), ...
    '9bc5ad0c5ed61945dc4282c36dfca7c789d3e0971f4af8c230145eb86099ccd5'));
T = load(fullfile(pd, 'test_rows.mat'), 'sT', 'pAll', 'taus');
B.sT = T.sT; B.taus = T.taus;
B.p = T.pAll; B.has = T.sT >= min(T.taus);
B.trig = find(ismember(B.names, c.trig))';
end
