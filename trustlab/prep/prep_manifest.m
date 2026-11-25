function prep_manifest()
pth = prep_paths();
RAW_DIR = pth.raw;
OUT_DIR = fullfile(pth.out, 'S01');
if ~exist(OUT_DIR, 'dir'); mkdir(OUT_DIR); end

files = dir(fullfile(RAW_DIR, '*.csv'));
if isempty(files)
    error('No CSV files found in %s', RAW_DIR);
end
nF = numel(files);

man = table('Size', [nF 10], ...
    'VariableTypes', {'string','double','string','double','double','double','string','double','double','string'}, ...
    'VariableNames', {'file','bytes','sha256','raw_lines','data_rows','n_cols', ...
                      'label_col','label_idx','n_unique_labels','note'});
schemaRows = strings(0,3);
labelRows  = strings(0,3);
allHeaders = cell(nF,1);

for i = 1:nF
    fp  = fullfile(files(i).folder, files(i).name);
    fnm = string(files(i).name);
    note = "";
    nbytes = files(i).bytes;
    sha = local_sha256(fp);
    rawLines = local_count_lines(fp);

    fid = fopen(fp, 'r');
    hdrLine = fgetl(fid);
    fclose(fid);
    hdr = strtrim(string(strsplit(hdrLine, ',')));
    nCols = numel(hdr);
    allHeaders{i} = hdr;
    for c = 1:nCols
        schemaRows(end+1, :) = [fnm, string(c), hdr(c)];
    end

    labIdx = find(contains(lower(hdr), "label"), 1);
    if isempty(labIdx)
        labName = "<none>"; labIdx = 0;
        note = note + "no Label column found; ";
    else
        labName = hdr(labIdx);
    end

    dataRows = NaN; nUniq = 0;
    if labIdx > 0
        try
            ds = tabularTextDatastore(fp, 'Delimiter', ',', 'TextType', 'string');
            vn = string(ds.VariableNames);
            if labIdx <= numel(vn)
                selName = vn(labIdx);
            else
                selName = vn(find(contains(lower(vn),"label"),1));
            end
            ds.SelectedVariableNames = cellstr(selName);
            T = readall(ds);
            lv = strtrim(string(T{:,1}));
            dataRows = numel(lv);
            [u, ~, g] = unique(lv);
            cnt = accumarray(g, 1);
            nUniq = numel(u);
            for k = 1:nUniq
                labelRows(end+1, :) = [fnm, u(k), string(cnt(k))];
            end
        catch ME
            note = note + "label parse failed (" + string(ME.identifier) + "); ";
        end
    end

    if ~isnan(dataRows) && rawLines ~= dataRows + 1
        note = note + sprintf("ROW MISMATCH raw=%d parsed=%d (check malformed/truncated); ", ...
                              rawLines, dataRows);
    end

    man(i,:) = {fnm, nbytes, sha, rawLines, dataRows, nCols, labName, labIdx, nUniq, strtrim(note)};
end

colCounts = cellfun(@numel, allHeaders);
colCounts = colCounts(:);
[uc, ~, gc] = unique(colCounts);
nPer = accumarray(gc, 1);
[~, modeIdx] = max(nPer);
modeCols = uc(modeIdx);
parts = strings(numel(uc), 1);
for k = 1:numel(uc)
    parts(k) = sprintf('%d cols in %d file(s)', uc(k), nPer(k));
end
schemaNote = sprintf('Column-count values across files: %s. Most common = %d.', ...
                     strjoin(parts', '; '), modeCols);

allNames = strings(0,1);
for i = 1:nF
    allNames = [allNames; allHeaders{i}(:)];
end
allNames = unique(allNames);
fileNames = string({files.name});
missReport = strings(0,1);
for n = 1:numel(allNames)
    present = false(nF,1);
    for i = 1:nF
        present(i) = any(allHeaders{i} == allNames(n));
    end
    if ~all(present)
        missReport(end+1,1) = allNames(n) + "  -> MISSING in: " + strjoin(fileNames(~present), ', ');
    end
end

writetable(man, fullfile(OUT_DIR, 'S01_manifest.csv'));
writematrix(["file","col_index","col_name"; schemaRows], fullfile(OUT_DIR, 'S01_schema.csv'));
writematrix(["file","raw_label","count";     labelRows ], fullfile(OUT_DIR, 'S01_labels.csv'));

fid = fopen(fullfile(OUT_DIR, 'S01_summary.txt'), 'w');
fprintf(fid, 'S01 - MANIFEST / SCHEMA / COUNTS (TRUSTLab)\n');
fprintf(fid, 'Generated: %s\n', datestr(now));
fprintf(fid, 'Source: %s\n', RAW_DIR);
fprintf(fid, 'Files found: %d\n\n', nF);

fprintf(fid, '== PER-FILE MANIFEST ==\n');
fprintf(fid, '%-28s %12s %8s %12s %12s %6s %-14s %5s\n', ...
        'file','bytes','cols','raw_lines','data_rows','uniq','label_col','lidx');
totRows = 0;
for i = 1:nF
    fprintf(fid, '%-28s %12d %8d %12d %12s %6d %-14s %5d\n', ...
        man.file(i), man.bytes(i), man.n_cols(i), man.raw_lines(i), ...
        num2str(man.data_rows(i)), man.n_unique_labels(i), man.label_col(i), man.label_idx(i));
    if ~isnan(man.data_rows(i)); totRows = totRows + man.data_rows(i); end
end
fprintf(fid, '\nTotal parsed data rows across all files: %d\n\n', totRows);

fprintf(fid, '== SCHEMA CONSISTENCY ==\n%s\n', schemaNote);
if isempty(missReport)
    fprintf(fid, 'All files share the same column names.\n\n');
else
    fprintf(fid, 'Columns NOT present in every file:\n');
    for r = 1:numel(missReport); fprintf(fid, '  %s\n', missReport(r)); end
    fprintf(fid, '\n');
end

fprintf(fid, '== LABEL VALUES (raw, per file) ==\n');
for i = 1:nF
    idx = labelRows(:,1) == man.file(i);
    if any(idx)
        subs = labelRows(idx, :);
        vals = strings(1, size(subs,1));
        for k = 1:size(subs,1)
            vals(k) = subs(k,2) + " (" + subs(k,3) + ")";
        end
        fprintf(fid, '  %-28s : %s\n', man.file(i), strjoin(vals, ', '));
    else
        fprintf(fid, '  %-28s : <no label read>\n', man.file(i));
    end
end

fprintf(fid, '\n== NOTES / WARNINGS ==\n');
anyNote = false;
for i = 1:nF
    if strlength(man.note(i)) > 0
        fprintf(fid, '  %-28s : %s\n', man.file(i), man.note(i)); anyNote = true;
    end
end
if ~anyNote; fprintf(fid, '  none\n'); end
fclose(fid);
end

function sha = local_sha256(fp)
md  = java.security.MessageDigest.getInstance('SHA-256');
fid = fopen(fp, 'r');
if fid < 0; sha = "<open failed>"; return; end
while true
    bytes = fread(fid, 1048576, '*uint8');
    if isempty(bytes); break; end
    md.update(bytes);
end
fclose(fid);
d = typecast(md.digest(), 'uint8');
sha = string(lower(sprintf('%02x', d)));
end

function n = local_count_lines(fp)
n = 0; fid = fopen(fp, 'r');
if fid < 0; n = -1; return; end
while true
    chunk = fread(fid, 4194304, '*uint8');
    if isempty(chunk); break; end
    n = n + sum(chunk == uint8(10));
end
fclose(fid);
fid = fopen(fp, 'r'); fseek(fid, -1, 'eof'); lastb = fread(fid, 1, '*uint8'); fclose(fid);
if ~isempty(lastb) && lastb ~= uint8(10); n = n + 1; end
end
