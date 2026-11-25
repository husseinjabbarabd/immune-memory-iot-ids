function tl_dist(f, dA, dB)
d = fileparts(f);
if ~exist(d, 'dir'); mkdir(d); end
save(f, 'dA', 'dB', '-v7');
end
