function [w, fb, hist] = tl_spec(K, wsel, c)
L = c.levels; nl = numel(L); F = numel(wsel);
mi = find(wsel > 0); nm = numel(mi);
rs = RandStream('twister', 'Seed', c.seed_spec);
P = ones(c.pop, F);
P(1, mi) = nl;
[~, P(2, :)] = ismember(wsel, L);
P(3:end, mi) = randi(rs, nl, c.pop - 2, nm);
fit = tl_evalw(L(P), K); ne = c.pop;
hist = zeros(c.gen_spec, 4);
for g = 1:c.gen_spec
    [fit, o] = sort(fit, 'descend'); P = P(o, :);
    fr = max(fit) - min(fit);
    if fr > 0; fn = (fit - min(fit)) / fr; else; fn = ones(c.pop, 1); end
    CL = []; own = [];
    for i = 1:c.pop
        nc = max(1, round(c.beta * c.pop / i));
        pm = c.pm0 * exp(-c.rho * fn(i));
        for j = 1:nc
            ci = P(i, :);
            mut = rand(rs, 1, nm) < pm;
            if ~any(mut); mut(randi(rs, nm)) = true; end
            for f = mi(mut)
                oth = setdiff(1:nl, ci(f)); ci(f) = oth(randi(rs, numel(oth)));
            end
            CL = [CL; ci]; own = [own; i];
        end
    end
    fc = tl_evalw(L(CL), K); ne = ne + numel(fc);
    for i = 1:c.pop
        j = find(own == i); [bf, b] = max(fc(j));
        if bf > fit(i); fit(i) = bf; P(i, :) = CL(j(b), :); end
    end
    [fit, o] = sort(fit, 'descend'); P = P(o, :);
    jj = c.pop - c.dnew + 1:c.pop;
    for j = jj; P(j, :) = 1; P(j, mi) = randi(rs, nl, 1, nm); end
    fit(jj) = tl_evalw(L(P(jj, :)), K); ne = ne + numel(jj);
    hist(g, :) = [g, max(fit), mean(fit), ne];
end
[fb, ib] = max(fit); w = L(P(ib, :));
end
