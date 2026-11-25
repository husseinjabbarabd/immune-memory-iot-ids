function prep_features()
pth    = prep_paths();
RAW    = pth.raw;
S08DIR = fullfile(pth.out, 'S08');
OUT    = fullfile(pth.out, 'S09');
READSZ = 200000;
if ~exist(OUT, 'dir'), mkdir(OUT); end

lowerKeys = ["api","benign","bruteforce","bufferoverflow","c2beac","ddos","dns","dos", ...
             "evasion","exfiltracion","explotacion","mitm","portscan","slowloris","tlsssl","webbased"];
metaNames = {'Src Port', 'Dst Port', 'Protocol'};

S = load(fullfile(S08DIR, 'S08_split.mat'), 'classId', 'fileId', 'rowInFile', 'split', ...
         'fileNames', 'classNames', 'hashSplit');
N         = numel(S.classId);
fileNames = string(S.fileNames);
nF        = numel(fileNames);
rowsPerFile = accumarray(double(S.fileId), 1, [nF 1]);

hdrs = cell(nF, 1);
for f = 1:nF
    hdrs{f} = read_header(fullfile(RAW, fileNames(f)));
end
common = hdrs{1};
for f = 2:nF
    common = intersect(common, hdrs{f}, 'stable');
end
h1 = hdrs{1};
flowNames = {};
for k = 7:numel(h1)
    nm = h1{k};
    if strcmp(nm, 'Label'), continue; end
    if ismember(nm, common), flowNames{end+1} = nm; end
end
if numel(flowNames) ~= 71
    error('Expected 71 common flow-statistic columns, found %d.', numel(flowNames));
end
featNames = [flowNames, metaNames];
nX = numel(featNames);
idxFlow  = 1:71;
idxPort  = 72:73;
idxProto = 74;

X   = zeros(N, nX, 'single');
ts  = zeros(N, 1, 'double');
XCW = nan(N, 2, 'single');
replaced = zeros(1, nX);
inexact  = zeros(1, nX);
tsBad    = 0;
fileRowsRead = zeros(nF, 1);
fileRepl = zeros(nF, 1);

pos = 0;
for f = 1:nF
    fpath = fullfile(RAW, fileNames(f));
    hdr   = hdrs{f};

    need = [featNames, {'Timestamp', 'Label'}];
    pn = zeros(1, numel(need));
    for k = 1:numel(need)
        j = find(strcmp(hdr, need{k}), 1);
        if isempty(j), error('Column "%s" missing in %s.', need{k}, fileNames(f)); end
        pn(k) = j;
    end
    pCWR = find(strcmp(hdr, 'CWR Flag Cnt'), 1);
    pCWE = find(strcmp(hdr, 'CWE Flag Count'), 1);

    ds = tabularTextDatastore(char(fpath));
    dsNames = ds.VariableNames;
    if numel(dsNames) ~= numel(hdr)
        error('Datastore found %d variables but header has %d in %s.', numel(dsNames), numel(hdr), fileNames(f));
    end
    selCols = sort(unique([pn, pCWR, pCWE]));
    fmts = repmat({'%f'}, 1, numel(selCols));
    fmts{selCols == pn(end)} = '%q';
    ds.SelectedVariableNames = dsNames(selCols);
    ds.SelectedFormats       = fmts;
    ds.ReadSize              = READSZ;

    fcount = 0;
    while hasdata(ds)
        t = read(ds);
        n = height(t);
        if pos + n > N, error('Row count exceeds N while reading %s.', fileNames(f)); end
        rr = pos+1:pos+n;

        lab = lower(strtrim(string(t.(dsNames{pn(end)}))));
        [tf, li] = ismember(lab, lowerKeys);
        if any(~tf) || any(uint8(li(:)) ~= S.classId(rr))
            error('Label/row alignment mismatch with S08 in %s near row %d.', fileNames(f), fcount+1);
        end
        if any(S.fileId(rr) ~= f)
            error('File id mismatch with S08 in %s.', fileNames(f));
        end

        for k = 1:nX
            v = double(t.(dsNames{pn(k)}));
            bad = ~isfinite(v);
            nb = sum(bad);
            if nb > 0
                replaced(k) = replaced(k) + nb;
                fileRepl(f) = fileRepl(f) + nb;
                v(bad) = 0;
            end
            vs = single(v);
            inexact(k) = inexact(k) + sum(double(vs) ~= v);
            X(rr, k) = vs;
        end
        tv = double(t.(dsNames{pn(nX+1)}));
        tsBad = tsBad + sum(~isfinite(tv));
        ts(rr) = tv;
        if ~isempty(pCWR), XCW(rr, 1) = single(double(t.(dsNames{pCWR}))); end
        if ~isempty(pCWE), XCW(rr, 2) = single(double(t.(dsNames{pCWE}))); end

        pos = pos + n;
        fcount = fcount + n;
    end
    fileRowsRead(f) = fcount;
    if fcount ~= rowsPerFile(f)
        error('File %s: read %d rows, S08 has %d.', fileNames(f), fcount, rowsPerFile(f));
    end
end
if pos ~= N, error('Total rows read %d != S08 N %d.', pos, N); end

cmin = zeros(1, nX); cmax = zeros(1, nX); nzero = zeros(1, nX);
for k = 1:nX
    col = X(:, k);
    cmin(k) = double(min(col)); cmax(k) = double(max(col)); nzero(k) = sum(col == 0);
end
grp = repmat("flow", nX, 1); grp(idxPort) = "port"; grp(idxProto) = "protocol";
colT = table((1:nX)', string(featNames(:)), grp, replaced(:), inexact(:), cmin(:), cmax(:), ...
    nzero(:), 100*nzero(:)/N, ...
    'VariableNames', {'idx','name','group','nonfinite_replaced','inexact_in_single','min','max','n_zero','pct_zero'});
writetable(colT, fullfile(OUT, 'S09_column_summary.csv'));

fT = table(fileNames(:), rowsPerFile, fileRowsRead, fileRepl, ...
    'VariableNames', {'file','rows_S08','rows_read','nonfinite_replaced'});
writetable(fT, fullfile(OUT, 'S09_file_rows.csv'));

md = java.security.MessageDigest.getInstance('SHA-256');
for k = 1:nX
    md.update(typecast(X(:, k), 'int8'));
end
hashX = lower(reshape(dec2hex(typecast(md.digest(), 'uint8'), 2)', 1, []));
md2 = java.security.MessageDigest.getInstance('SHA-256');
md2.update(typecast(ts, 'int8'));
hashTs = lower(reshape(dec2hex(typecast(md2.digest(), 'uint8'), 2)', 1, []));

classId = S.classId; split = S.split; fileId = S.fileId; rowInFile = S.rowInFile;
classNames = S.classNames; hashSplit = S.hashSplit;
save(fullfile(OUT, 'S09_features.mat'), 'X', 'ts', 'XCW', 'featNames', 'idxFlow', 'idxPort', ...
     'idxProto', 'classId', 'split', 'fileId', 'rowInFile', 'classNames', 'fileNames', ...
     'hashSplit', 'hashX', 'hashTs', '-v7.3');

fid = fopen(fullfile(OUT, 'S09_summary.txt'), 'w');
pr = @(varargin) fprintf(fid, varargin{:});
pr('S09 - Feature matrix\n');
pr('====================\n');
pr('Rows                         : %d (= S08 N; label/file alignment verified for every row)\n', N);
pr('X columns                    : %d  (71 flow stats [common by name] + Src Port + Dst Port + Protocol)\n', nX);
pr('Timestamp                    : separate double vector (finite: %d of %d)\n', N - tsBad, N);
pr('Timestamp range (epoch s)    : %.3f .. %.3f\n', min(ts), max(ts));
pr('CWR/CWE                      : not in X (stored in XCW as [CWR CWE], NaN = absent)\n');
pr('Preprocessing applied        : NaN/Inf -> 0 only.\n');
pr('NaN/Inf replaced (total)     : %d cells (per file in S09_file_rows.csv, total %d)\n', sum(replaced), sum(fileRepl));
pr('Columns with replaced cells  : %d\n', sum(replaced > 0));
pr('Cells changed by single()    : %d across %d columns (see inexact_in_single)\n', sum(inexact), sum(inexact > 0));
pr('SHA-256 X (col-wise)         : %s\n', hashX);
pr('SHA-256 Timestamp            : %s\n', hashTs);
pr('SHA-256 split vector (S08)   : %s\n', hashSplit);
pr('Index sets saved             : idxFlow = 1:71, idxPort = 72:73, idxProto = 74\n');
pr('\nPer-file rows: all equal to S08 counts\n');
fclose(fid);
end

function hdr = read_header(path)
fid = fopen(char(path), 'r');
line = fgetl(fid);
fclose(fid);
line = regexprep(line, '^[^\x00-\x7F]+', '');
parts = strsplit(line, ',', 'CollapseDelimiters', false);
hdr = strtrim(strrep(parts, '"', ''));
end
