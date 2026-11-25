function [p, st] = tl_p2(codeTr, yTr, X, wM, wE, trig, nC, c)
[U, ~, g] = unique(codeTr, 'rows');
[~, labU] = max(accumarray([g, yTr], 1, [size(U, 1), nC]), [], 2);
st.mem = size(U, 1); st.memhash = tl_sha([typecast(U(:), 'uint8'); uint8(labU(:))]);
[tf, loc] = ismember(X, U, 'rows');
p = zeros(size(X, 1), 1); p(tf) = labU(loc(tf));
iu = find(~tf);
[D1, I1] = tl_knn(X(iu, :), U, wM, c.k, c);
v = vote(labU(I1), c.k, nC); d1 = D1(:, 1) / (1023^2 * sum(wM));
p(iu) = v;
ic = ismember(v, trig); iC = iu(ic);
[DE, IE] = tl_knn(X(iC, :), U, wE, c.k, c);
vE = vote(labU(IE), c.k, nC); dE = DE(:, 1) / (1023^2 * sum(wE));
dis = vE ~= p(iC); take = dis & (dE < d1(ic));
p(iC(take)) = vE(take);
st.matched = sum(tf); st.trig = numel(iC); st.dis = sum(dis); st.take = sum(take);
end

function p = vote(L, k, nC)
n = size(L, 1); cnt = zeros(n, nC);
for j = 1:k
    li = sub2ind([n nC], (1:n)', L(:, j)); cnt(li) = cnt(li) + 1;
end
mx = max(cnt, [], 2); p = zeros(n, 1);
for j = k:-1:1
    ok = cnt(sub2ind([n nC], (1:n)', L(:, j))) == mx; p(ok) = L(ok, j);
end
end
