function [D, r, q] = tl_frame(code, y, cls, benign, c)
rs = RandStream('twister', 'Seed', c.seed_frame);
r = []; q = [];
for k = cls(:)'
    rows = find(y == k);
    if k == benign; nr = c.nr_ben; nq = c.nq_ben; else; nr = c.nr_att; nq = c.nq_att; end
    p = rows(randperm(rs, numel(rows), nr + nq));
    r = [r; p(1:nr)]; q = [q; p(nr+1:end)];
end
map = zeros(max(y), 1); map(cls) = 1:numel(cls);
D.XR = single(code(r, :)) / 1023; D.yR = map(y(r));
D.XQ = single(code(q, :)) / 1023; D.yQ = map(y(q));
D.nc = numel(cls); D.tar = 0; D.qblock = c.qblock;
D.CR = code(r, :); D.CQ = code(q, :); D.cells = 2.5e7;
D.id = tl_sha([typecast(D.CR(:), 'uint8'); typecast(D.CQ(:), 'uint8'); uint8(D.yR); uint8(D.yQ)]);
end
