function m = tl_map(Xtr)
F = size(Xtr, 2);
m.u = cell(F, 1); m.less = cell(F, 1); m.cnt = cell(F, 1); m.n = size(Xtr, 1);
for j = 1:F
    [u, ~, ic] = unique(double(Xtr(:, j)));
    cnt = accumarray(ic, 1);
    m.u{j} = u; m.cnt{j} = cnt; m.less{j} = cumsum(cnt) - cnt;
end
end
