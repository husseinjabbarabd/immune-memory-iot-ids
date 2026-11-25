function prep_split()
pth    = prep_paths();
RAW    = pth.raw;
OUT    = fullfile(pth.out, 'S08');
SEED   = 20260401;
P_TEST = 0.20;
P_VAL  = 0.20;
READSZ = 500000;
if ~exist(OUT, 'dir'), mkdir(OUT); end

classNames = ["API","Benign","Bruteforce","BufferOverflow","C2Beaconing","DDoS","DNS","DoS", ...
              "Evasion","Exfiltration","Exploitation","MITM","PortScan","Slowloris","TLSSSL","WebBased"];
lowerKeys  = ["api","benign","bruteforce","bufferoverflow","c2beac","ddos","dns","dos", ...
              "evasion","exfiltracion","explotacion","mitm","portscan","slowloris","tlsssl","webbased"];
expected     = [158575 2572388 105712 92326 118651 188820 168866 149999 ...
              85422 111752 150576 106148 107566 108785 122145 153947];
nCls = numel(classNames);
N    = sum(expected);

d = dir(fullfile(RAW, '*.csv'));
fileNames = sort(string({d.name}));
nF = numel(fileNames);
if nF ~= 16
    error('Expected 16 CSV files in %s, found %d.', RAW, nF);
end

classId   = zeros(N, 1, 'uint8');
fileId    = zeros(N, 1, 'uint8');
rowInFile = zeros(N, 1, 'uint32');
FC = zeros(nF, nCls);
pos = 0;
for f = 1:nF
    fpath = fullfile(RAW, fileNames(f));
    ds = tabularTextDatastore(char(fpath));
    ds.SelectedVariableNames = {'Label'};
    ds.SelectedFormats       = {'%q'};
    ds.ReadSize              = READSZ;
    rowCounter = 0;
    while hasdata(ds)
        t   = read(ds);
        lab = lower(strtrim(string(t.Label)));
        [tf, idx] = ismember(lab, lowerKeys);
        if any(~tf)
            bad = unique(lab(~tf));
            error('Unknown/missing label(s) in %s: %s', fileNames(f), strjoin(bad(1:min(5,numel(bad))), ', '));
        end
        n = numel(idx);
        if pos + n > N
            error('Row count exceeds expected total (%d) while reading %s.', N, fileNames(f));
        end
        classId(pos+1:pos+n)   = uint8(idx);
        fileId(pos+1:pos+n)    = uint8(f);
        rowInFile(pos+1:pos+n) = uint32(rowCounter + (1:n)');
        FC(f, :) = FC(f, :) + accumarray(idx(:), 1, [nCls 1])';
        pos = pos + n;
        rowCounter = rowCounter + n;
    end
end

if pos ~= N
    error('Total rows read = %d, expected %d.', pos, N);
end
readCounts = accumarray(double(classId), 1, [nCls 1])';
if any(readCounts ~= expected)
    error('Per-class counts differ from the expected counts: %s', strjoin(classNames(readCounts ~= expected), ', '));
end

rng(SEED, 'twister');
split = zeros(N, 1, 'uint8');
for c = 1:nCls
    idx   = find(classId == c);
    n     = numel(idx);
    p     = idx(randperm(n));
    nTest = round(P_TEST * n);
    nDev  = n - nTest;
    nVal  = round(P_VAL * nDev);
    split(p(1:nTest))                 = 3;
    split(p(nTest+1:nTest+nVal))      = 2;
    split(p(nTest+nVal+1:end))        = 1;
end

if any(split == 0), error('Some rows were not assigned.'); end
T = accumarray([double(classId) double(split)], 1, [nCls 3]);
if any(sum(T, 2)' ~= expected), error('Split counts do not add up to class totals.'); end
if sum(T(:)) ~= N, error('Split total differs from N.'); end

FS = zeros(nF, 3);
for f = 1:nF
    FS(f, :) = accumarray(double(split(fileId == f)), 1, [3 1])';
end

hashSplit = sha_vec(split);
hashClass = sha_vec(classId);

splitNames = ["Train","Validation","Test"];
save(fullfile(OUT, 'S08_split.mat'), 'classId', 'fileId', 'rowInFile', 'split', ...
     'classNames', 'fileNames', 'splitNames', 'SEED', 'P_TEST', 'P_VAL', 'hashSplit', 'hashClass', '-v7.3');

tbl = table(classNames(:), expected(:), T(:,1), T(:,2), T(:,3), ...
    100*T(:,3)./expected(:), 100*T(:,2)./expected(:), 100*T(:,1)./expected(:), ...
    'VariableNames', {'class','total','train','validation','test','test_pct','val_pct','train_pct'});
tot = table("ALL", sum(expected), sum(T(:,1)), sum(T(:,2)), sum(T(:,3)), ...
    100*sum(T(:,3))/N, 100*sum(T(:,2))/N, 100*sum(T(:,1))/N, ...
    'VariableNames', tbl.Properties.VariableNames);
tbl = [tbl; tot];
writetable(tbl, fullfile(OUT, 'S08_split_counts.csv'));

fcT = array2table(FC, 'VariableNames', cellstr(classNames));
fcT = addvars(fcT, fileNames(:), 'Before', 1, 'NewVariableNames', 'file');
writetable(fcT, fullfile(OUT, 'S08_file_class_matrix.csv'));

fsT = table(fileNames(:), FS(:,1), FS(:,2), FS(:,3), 'VariableNames', {'file','train','validation','test'});
writetable(fsT, fullfile(OUT, 'S08_file_split_matrix.csv'));

fid = fopen(fullfile(OUT, 'S08_summary.txt'), 'w');
pr = @(varargin) fprintf(fid, varargin{:});
pr('S08 - Stratified random split\n');
pr('=============================\n');
pr('Rows assigned           : %d (expected total %d)\n', N, N);
pr('Seed                    : %d (twister)\n', SEED);
pr('Test share per class    : %.0f%%\n', 100*P_TEST);
pr('Validation share of rest: %.0f%%\n', 100*P_VAL);
pr('Train / Val / Test rows : %d / %d / %d\n', sum(T(:,1)), sum(T(:,2)), sum(T(:,3)));
pr('SHA-256 split vector    : %s\n', hashSplit);
pr('SHA-256 class vector    : %s\n', hashClass);
pr('\nPer-class counts (Train / Validation / Test):\n');
for c = 1:nCls
    pr('  %-15s total %8d | train %8d | val %7d | test %7d\n', classNames(c), expected(c), T(c,1), T(c,2), T(c,3));
end
pr('\nBenign test rows: %d | Attack test rows: %d\n', T(2,3), sum(T(:,3)) - T(2,3));
pr('\nChecks: every row assigned once; per-class counts equal the expected counts.\n');
fclose(fid);

end
