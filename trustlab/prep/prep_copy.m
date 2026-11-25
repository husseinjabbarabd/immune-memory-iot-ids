function prep_copy()
pth = prep_paths();
out = fullfile(pth.fs, 'FS_S00');
if ~exist(out, 'dir'); mkdir(out); end
src = {fullfile(pth.sup, 'SUP_S00', 'S09_features.mat'), ...
       fullfile(pth.sup, 'SUP_S01', 'SUP_S01_split.mat'), ...
       fullfile(pth.sup, 'SUP_S00', 'SUP_class_names.mat')};

fid = fopen(fullfile(out, 'FS_S00_input_hashes.txt'), 'w');
for k = 1:numel(src)
    [~, nm, ex] = fileparts(src{k}); nm = [nm ex];
    [ok, msg] = copyfile(src{k}, fullfile(out, nm));
    assert(ok, 'Copy failed: %s', msg);
    h = sha_file(src{k});
    assert(strcmp(h, sha_file(fullfile(out, nm))), 'Copy of %s differs from its source', nm);
    fprintf(fid, '%s  %s\n', h, nm);
end
fclose(fid);
end
