function t = tl_match(s, far)
n = numel(s); K = round(far * n);
if K < 1; t = Inf; return; end
sd = sort(s(:), 'descend'); v = sd(min(K, n));
if sum(s >= v) <= K; t = v; return; end
hi = sd(sd > v);
if isempty(hi); t = Inf; else; t = min(hi); end
end
