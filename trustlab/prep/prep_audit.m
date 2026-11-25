function prep_audit()
pth = prep_paths();
RAW_DIR = pth.raw;
OUT_DIR = fullfile(pth.out, 'S02');
if ~exist(OUT_DIR, 'dir'); mkdir(OUT_DIR); end

files = dir(fullfile(RAW_DIR, '*.csv'));
if isempty(files); error('No CSV files found in %s', RAW_DIR); end
nF = numel(files);

mapKeys = {'api','benign','bruteforce','bufferoverflow','c2beac','c2beaconing', ...
           'ddos','dns','dos','evasion','exfiltracion','exfiltration', ...
           'explotacion','exploitation','mitm','portscan','slowloris', ...
           'tlsssl','webbased'};
mapVals = {'API','Benign','Bruteforce','BufferOverflow','C2Beaconing','C2Beaconing', ...
           'DDoS','DNS','DoS','Evasion','Exfiltration','Exfiltration', ...
           'Exploitation','Exploitation','MITM','PortScan','Slowloris', ...
           'TLSSSL','WebBased'};
labelMap = containers.Map(mapKeys, mapVals);

expKeys = {'API','Benign','Bruteforce','BufferOverflow','C2Beaconing','DDoS','DNS', ...
           'DoS','Evasion','Exfiltration','Exploitation','MITM','PortScan', ...
           'Slowloris','TLSSSL','WebBased'};
expVals = [158575 2572388 105712 92326 118651 188820 168866 149999 85422 ...
           111752 150576 106148 107566 108785 122145 153947];
expCount = containers.Map(expKeys, expVals);

interest = {'CWE Flag Count','CWR Flag Cnt','ECE Flag Cnt','Fwd Header Len','Bwd Header Len'};

allHdr   = cell(nF,1);
fileCls  = strings(nF,1);
labRows  = {};
statRows = {};
fcRows   = {};
badRows  = {};
classTot = containers.Map('KeyType','char','ValueType','double');
fileMismatch = zeros(nF,1);
fileRows     = zeros(nF,1);
benLabSrc = containers.Map('KeyType','char','ValueType','double');
benRel = struct('n',0,'nEq',0,'nCWEnz',0,'nCWRnz',0,'nBothnz',0,'has',false);

for i = 1:nF
    fp  = fullfile(files(i).folder, files(i).name);
    fnm = string(files(i).name);
    base = lower(regexprep(char(fnm), '\.csv$', ''));
    base = regexprep(base, '_recovered$', '');
    if isKey(labelMap, base); fileCls(i) = string(labelMap(base)); else; fileCls(i) = "UNKNOWN"; end

    fid = fopen(fp,'r'); hdrLine = fgetl(fid); fclose(fid);
    hdr = strtrim(string(strsplit(hdrLine, ',')));
    nCols = numel(hdr);
    allHdr{i} = hdr;

    fid = fopen(fp,'r');
    fgetl(fid);
    lineNo = 1; nBad = 0; nEmpty = 0; nLines = 0;
    while true
        ln = fgetl(fid);
        if ~ischar(ln); break; end
        lineNo = lineNo + 1;
        if isempty(ln); nEmpty = nEmpty + 1; continue; end
        nLines = nLines + 1;
        nf = nnz(ln == ',') + 1;
        if nf ~= nCols
            nBad = nBad + 1;
            if nBad <= 25
                badRows(end+1,:) = {char(fnm), lineNo, nf, nCols, ln(1:min(numel(ln),160))};
            end
        end
    end
    fclose(fid);
    fcRows(end+1,:) = {char(fnm), nLines, nCols, nBad, nEmpty};

    labIdx = find(lower(hdr) == "label", 1);
    srcIdx = find(hdr == "Source File", 1);
    selIdx = labIdx;
    if ~isempty(srcIdx); selIdx(end+1) = srcIdx; end
    numNames = strings(0,1);
    for k = 1:numel(interest)
        j = find(hdr == string(interest{k}), 1);
        if ~isempty(j); selIdx(end+1) = j; numNames(end+1,1) = hdr(j); end
    end
    selIdx = unique(selIdx);

    ds = tabularTextDatastore(fp, 'Delimiter', ',', 'TextType', 'string', 'ReadSize', 200000);
    vn = string(ds.VariableNames);
    ds.SelectedVariableNames = cellstr(vn(selIdx));
    fm = cell(1, numel(selIdx));
    for k = 1:numel(selIdx)
        if selIdx(k) == labIdx || (~isempty(srcIdx) && selIdx(k) == srcIdx)
            fm{k} = '%q';
        else
            fm{k} = '%f';
        end
    end
    ds.SelectedFormats = fm;

    labCnt = containers.Map('KeyType','char','ValueType','double');
    st = struct();
    for k = 1:numel(numNames); st.(matlab.lang.makeValidName(numNames(k))) = newstat(); end
    hasCWE = any(numNames == "CWE Flag Count");
    hasCWR = any(numNames == "CWR Flag Cnt");

    while hasdata(ds)
        T = read(ds);
        lab = strtrim(string(T.(char(vn(labIdx)))));
        [u,~,g] = unique(lab); c = accumarray(g,1);
        for q = 1:numel(u)
            labCnt = bump(labCnt, char(u(q)), c(q));
        end
        if ~isempty(srcIdx)
            src = strtrim(string(T.(char(vn(srcIdx)))));
            [u2,~,g2] = unique(lab + "||" + src); c2 = accumarray(g2,1);
            for q = 1:numel(u2)
                benLabSrc = bump(benLabSrc, char(u2(q)), c2(q));
            end
        end
        for k = 1:numel(numNames)
            fld = matlab.lang.makeValidName(numNames(k));
            x = double(T.(char(vn(find(hdr == numNames(k), 1)))));
            st.(fld) = upstat(st.(fld), x);
        end
        if hasCWE && hasCWR
            a = double(T.(char(vn(find(hdr == "CWE Flag Count", 1)))));
            b = double(T.(char(vn(find(hdr == "CWR Flag Cnt", 1)))));
            benRel.has = true;
            benRel.n      = benRel.n + numel(a);
            benRel.nEq    = benRel.nEq + sum(a == b);
            benRel.nCWEnz = benRel.nCWEnz + sum(a ~= 0 & ~isnan(a));
            benRel.nCWRnz = benRel.nCWRnz + sum(b ~= 0 & ~isnan(b));
            benRel.nBothnz = benRel.nBothnz + sum(a ~= 0 & b ~= 0 & ~isnan(a) & ~isnan(b));
        end
    end

    keys = keys_(labCnt);
    nTot = 0; nMis = 0;
    for q = 1:numel(keys)
        raw = keys{q}; cnt = labCnt(raw);
        lk = lower(raw);
        if isKey(labelMap, lk); canon = labelMap(lk); else; canon = 'UNMAPPED'; end
        labRows(end+1,:) = {char(fnm), raw, canon, cnt};
        nTot = nTot + cnt;
        if ~strcmp(canon, char(fileCls(i))); nMis = nMis + cnt; end
        if isKey(classTot, canon); classTot(canon) = classTot(canon) + cnt; else; classTot(canon) = cnt; end
    end
    fileRows(i) = nTot; fileMismatch(i) = nMis;

    for k = 1:numel(numNames)
        s = st.(matlab.lang.makeValidName(numNames(k)));
        statRows(end+1,:) = {char(fnm), char(numNames(k)), s.n, s.nNaN, s.nInf, s.nZero, ...
            s.mn, s.mx, s.sum / max(1, s.n - s.nNaN - s.nInf), uniqStr(s)};
    end
end

sig = strings(nF,1);
for i = 1:nF; sig(i) = strjoin(allHdr{i}, '|'); end
[~, ~, gid] = unique(sig, 'stable');
nG = max(gid);

grpRows = {};
for i = 1:nF
    grpRows(end+1,:) = {char(files(i).name), gid(i), numel(allHdr{i})};
end

allNames = strings(0,1);
for i = 1:nF; allNames = [allNames; allHdr{i}(:)]; end
allNames = unique(allNames, 'stable');
comRows = {};
nCommon = 0;
for n = 1:numel(allNames)
    pos = zeros(1,nF);
    for i = 1:nF
        j = find(allHdr{i} == allNames(n), 1);
        if ~isempty(j); pos(i) = j; end
    end
    present = nnz(pos > 0);
    moved = numel(unique(pos(pos > 0))) > 1;
    comRows(end+1,:) = {char(allNames(n)), present, moved, strjoin(string(pos), ' ')};
    if present == nF; nCommon = nCommon + 1; end
end

write_csv(fullfile(OUT_DIR,'S02_label_map.csv'), {'file','raw_label','canonical','count'}, labRows);

ck = keys_(classTot);
ccRows = {};
for q = 1:numel(ck)
    cnt = classTot(ck{q});
    if isKey(expCount, ck{q}); e = expCount(ck{q}); else; e = NaN; end
    ccRows(end+1,:) = {ck{q}, cnt, e, cnt - e};
end
write_csv(fullfile(OUT_DIR,'S02_class_counts.csv'), {'canonical_class','rows','expected_rows','diff'}, ccRows);
copyfile(fullfile(OUT_DIR,'S02_class_counts.csv'), pth.counts);
write_csv(fullfile(OUT_DIR,'S02_schema_groups.csv'), {'file','header_group','n_cols'}, grpRows);
write_csv(fullfile(OUT_DIR,'S02_common_schema.csv'), {'column_name','n_files_present','position_differs_between_files','position_in_each_file'}, comRows);
write_csv(fullfile(OUT_DIR,'S02_column_stats.csv'), {'file','column','n','n_NaN','n_Inf','n_zero','min','max','mean_finite','distinct_values'}, statRows);
write_csv(fullfile(OUT_DIR,'S02_field_count_check.csv'), {'file','data_lines','header_cols','lines_with_wrong_field_count','empty_lines'}, fcRows);
write_csv(fullfile(OUT_DIR,'S02_bad_lines.csv'), {'file','line_no','fields_found','fields_expected','line_prefix'}, badRows);

bk = keys_(benLabSrc);
blRows = {};
for q = 1:numel(bk)
    parts = strsplit(bk{q}, '||');
    blRows(end+1,:) = {parts{1}, strjoin(parts(2:end), '||'), benLabSrc(bk{q})};
end
write_csv(fullfile(OUT_DIR,'S02_benign_label_x_sourcefile.csv'), {'raw_label','source_file','count'}, blRows);

fid = fopen(fullfile(OUT_DIR,'S02_summary.txt'),'w');
fprintf(fid, 'S02 - LABELS / SCHEMA / BENIGN AUDIT (TRUSTLab)\n');
fprintf(fid, 'Generated: %s\nSource: %s\nFiles: %d\n\n', datestr(now), RAW_DIR, nF);

fprintf(fid, '== Q1. LABEL CANONICALISATION ==\n');
fprintf(fid, '%-24s %-16s %12s %14s\n', 'file','expected_class','rows','label!=class');
for i = 1:nF
    fprintf(fid, '%-24s %-16s %12d %14d\n', files(i).name, fileCls(i), fileRows(i), fileMismatch(i));
end
fprintf(fid, '\nCanonical class counts (observed vs expected):\n');
fprintf(fid, '%-16s %12s %12s %8s\n', 'class','observed','expected','diff');
totObs = 0; totExp = 0;
for q = 1:size(ccRows,1)
    fprintf(fid, '%-16s %12d %12d %8d\n', ccRows{q,1}, ccRows{q,2}, ccRows{q,3}, ccRows{q,4});
    totObs = totObs + ccRows{q,2}; totExp = totExp + ccRows{q,3};
end
fprintf(fid, '%-16s %12d %12d %8d\n', 'TOTAL', totObs, totExp, totObs - totExp);
nUnm = 0; for q = 1:size(labRows,1); if strcmp(labRows{q,3},'UNMAPPED'); nUnm = nUnm + 1; end; end
fprintf(fid, 'Unmapped raw label values: %d\n', nUnm);
fprintf(fid, 'Total rows whose label disagrees with the file class: %d\n', sum(fileMismatch));
fprintf(fid, 'Raw label variants per class (see S02_label_map.csv):\n');
for q = 1:size(labRows,1)
    fprintf(fid, '  %-22s %-18s -> %-14s %10d\n', labRows{q,1}, labRows{q,2}, labRows{q,3}, labRows{q,4});
end

fprintf(fid, '\n== Q2. ROW INTEGRITY (field count per line vs header) ==\n');
fprintf(fid, '%-24s %12s %6s %10s %8s\n', 'file','data_lines','cols','badFields','empty');
for q = 1:size(fcRows,1)
    fprintf(fid, '%-24s %12d %6d %10d %8d\n', fcRows{q,1}, fcRows{q,2}, fcRows{q,3}, fcRows{q,4}, fcRows{q,5});
end
fprintf(fid, 'Lines with wrong field count (first 25 per file listed in S02_bad_lines.csv): %d\n', sum([fcRows{:,4}]));

fprintf(fid, '\n== Q3. SCHEMA GROUPS ==\n');
fprintf(fid, 'Distinct header layouts: %d\n', nG);
for g = 1:nG
    m = find(gid == g);
    fprintf(fid, '  group %d (%d cols): %s\n', g, numel(allHdr{m(1)}), strjoin(string({files(m).name}), ', '));
end
fprintf(fid, 'Columns present in ALL %d files (by name): %d\n', nF, nCommon);
fprintf(fid, 'Columns NOT in every file, or at different positions in different files:\n');
for q = 1:size(comRows,1)
    if comRows{q,2} < nF || comRows{q,3}
        fprintf(fid, '  %-24s present_in=%2d  position_differs=%d  positions=[%s]\n', ...
                comRows{q,1}, comRows{q,2}, comRows{q,3}, comRows{q,4});
    end
end
fprintf(fid, 'Columns must be selected by name, not by position.\n');

fprintf(fid, '\n== Q4. VALUE CHECK OF RENAMED / DISPLACED COLUMNS ==\n');
fprintf(fid, '%-22s %-16s %10s %10s %10s %12s %12s\n', 'file','column','nZero','nNaN','nInf','min','max');
for q = 1:size(statRows,1)
    fprintf(fid, '%-22s %-16s %10d %10d %10d %12.4g %12.4g   distinct: %s\n', ...
        statRows{q,1}, statRows{q,2}, statRows{q,6}, statRows{q,4}, statRows{q,5}, ...
        statRows{q,7}, statRows{q,8}, statRows{q,10});
end

fprintf(fid, '\n== Q5. BENIGN: LABEL x SOURCE FILE, AND CWE vs CWR ==\n');
fprintf(fid, 'Distinct (label, Source File) pairs: %d  (full table: S02_benign_label_x_sourcefile.csv)\n', size(blRows,1));
if ~isempty(blRows)
    cnts = [blRows{:,3}];
    [~, ord] = sort(cnts, 'descend');
    for q = 1:min(30, numel(ord))
        fprintf(fid, '  %-10s | %-60s | %10d\n', blRows{ord(q),1}, blRows{ord(q),2}, blRows{ord(q),3});
    end
    labs = unique(string(blRows(:,1)));
    for q = 1:numel(labs)
        m = string(blRows(:,1)) == labs(q);
        fprintf(fid, '  label %-10s : %d distinct Source File values, %d rows\n', labs(q), nnz(m), sum([blRows{m,3}]));
    end
end
if benRel.has
    fprintf(fid, 'Benign flag columns: rows=%d | CWE==CWR in %d rows (%.4f) | CWE!=0 in %d | CWR!=0 in %d | both!=0 in %d\n', ...
        benRel.n, benRel.nEq, benRel.nEq / benRel.n, benRel.nCWEnz, benRel.nCWRnz, benRel.nBothnz);
end
fclose(fid);
end

function m = bump(m, key, c)
if isKey(m, key); m(key) = m(key) + c; else; m(key) = c; end
end

function k = keys_(m)
k = m.keys();
k = sort(k);
end

function s = newstat()
s = struct('n',0,'nNaN',0,'nInf',0,'nZero',0,'mn',Inf,'mx',-Inf,'sum',0, ...
           'uv',containers.Map('KeyType','double','ValueType','double'),'over',false);
end

function s = upstat(s, x)
x = double(x(:));
s.n = s.n + numel(x);
nanm = isnan(x); infm = isinf(x);
s.nNaN = s.nNaN + sum(nanm);
s.nInf = s.nInf + sum(infm);
fin = x(~nanm & ~infm);
s.nZero = s.nZero + sum(fin == 0);
if ~isempty(fin)
    s.mn = min(s.mn, min(fin));
    s.mx = max(s.mx, max(fin));
    s.sum = s.sum + sum(fin);
    if ~s.over
        [u,~,g] = unique(fin); c = accumarray(g,1);
        for q = 1:numel(u)
            if isKey(s.uv, u(q))
                s.uv(u(q)) = s.uv(u(q)) + c(q);
            elseif s.uv.Count < 30
                s.uv(u(q)) = c(q);
            else
                s.over = true; break;
            end
        end
    end
end
end

function t = uniqStr(s)
if s.over
    t = 'many (>30)';
else
    k = cell2mat(s.uv.keys());
    parts = strings(1, numel(k));
    for q = 1:numel(k)
        parts(q) = sprintf('%g:%d', k(q), s.uv(k(q)));
    end
    t = char(strjoin(parts, '; '));
end
end

function write_csv(fp, hdr, rows)
fid = fopen(fp, 'w');
fprintf(fid, '%s\n', strjoin(hdr, ','));
for r = 1:size(rows,1)
    parts = cell(1, size(rows,2));
    for c = 1:size(rows,2)
        v = rows{r,c};
        if isnumeric(v) || islogical(v)
            parts{c} = sprintf('%.10g', v);
        else
            parts{c} = ['"' strrep(char(string(v)), '"', '""') '"'];
        end
    end
    fprintf(fid, '%s\n', strjoin(parts, ','));
end
fclose(fid);
end
