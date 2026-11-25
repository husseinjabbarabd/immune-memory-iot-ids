function p = prep_paths()
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
j = jsondecode(fileread(fullfile(root, 'paths.json'))); t = j.trustlab;
p.raw    = fullfile(root, t.raw);
p.out    = fullfile(root, t.out);
p.sup    = fullfile(root, t.sup);
p.fs     = fullfile(root, t.fs);
p.counts = fullfile(root, t.counts);
end
