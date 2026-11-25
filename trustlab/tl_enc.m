function C = tl_enc(X, m, cols)
if nargin < 3; cols = 1:size(X, 2); end
C = zeros(size(X, 1), numel(cols), 'uint16');
for jj = 1:numel(cols)
    j = cols(jj);
    x = double(X(:, jj));
    u = m.u{j};
    k = discretize(x, [-Inf; u(:); Inf]) - 1;
    F = zeros(size(x)); pos = k > 0; kk = k(pos);
    ex = x(pos) == u(kk);
    Fp = (m.less{j}(kk) + m.cnt{j}(kk)) / m.n;
    Fp(ex) = (m.less{j}(kk(ex)) + 0.5 * m.cnt{j}(kk(ex))) / m.n;
    F(pos) = Fp;
    C(:, jj) = uint16(round(1023 * F));
end
end
