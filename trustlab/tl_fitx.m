function [f, p] = tl_fitx(w, D)
w = single(w(:)') / 8; nz = w > 0;
if ~any(nz); f = 0; p = []; return; end
w = w(nz);
[~, ir] = unique(D.CR(:, nz), 'rows', 'stable');
[~, iq, jq] = unique(D.CQ(:, nz), 'rows');
R = D.XR(ir, nz); yR = D.yR(ir); Qa = D.XQ(iq, nz);
RW = R .* w; rn = sum(R.^2 .* w, 2)';
nQ = size(Qa, 1); pu = zeros(nQ, 1);
qb = max(1, floor(D.cells / numel(ir)));
for s = 1:qb:nQ
    e = min(s + qb - 1, nQ); Q = Qa(s:e, :);
    [~, i] = min(sum(Q.^2 .* w, 2) + rn - 2 * (Q * RW'), [], 2);
    pu(s:e) = yR(i);
end
p = pu(jq);
cm = accumarray([D.yQ, p], 1, [D.nc D.nc]);
f1 = zeros(D.nc, 1);
for k = 1:D.nc
    tp = cm(k, k); P = tp / max(1, sum(cm(:, k))); Rc = tp / max(1, sum(cm(k, :)));
    if P + Rc > 0; f1(k) = 2 * P * Rc / (P + Rc); end
end
if D.tar > 0; f = f1(D.tar); else; f = mean(f1); end
end
