function S = tl_clonalg(K, F, c)
kl = c.klist; kl(1) = F; nk = numel(kl); L = c.levels; nl = numel(L);
S.klist = kl; S.w = zeros(nk, F); S.f = zeros(nk, 1); S.hist = cell(nk, 1); S.loo = [];
for ik = 1:nk
    k = kl(ik);
    rs = RandStream('twister', 'Seed', c.seed_evo + k);
    P = zeros(c.pop, F);
    if ik == 1
        P(1, :) = nl;
        P(2:end, :) = randi(rs, nl, c.pop - 1, F);
    else
        P(1, :) = project(lev(S.w(ik - 1, :), L), k, S.loo);
        P(2, :) = project(lev(S.w(1, :), L), k, S.loo);
        for i = 3:c.pop; P(i, :) = feasible(rs, F, nl, k); end
    end
    fit = tl_evalw(L(P), K); ne = c.pop;
    hist = zeros(c.gen, 4);
    for g = 1:c.gen
        [fit, o] = sort(fit, 'descend'); P = P(o, :);
        fr = max(fit) - min(fit);
        if fr > 0; fn = (fit - min(fit)) / fr; else; fn = ones(c.pop, 1); end
        CL = []; own = [];
        for i = 1:c.pop
            nc = max(1, round(c.beta * c.pop / i));
            pm = c.pm0 * exp(-c.rho * fn(i));
            for j = 1:nc
                ci = P(i, :);
                mut = rand(rs, 1, F) < pm;
                if ~any(mut); mut(randi(rs, F)) = true; end
                for f = find(mut)
                    oth = setdiff(1:nl, ci(f)); ci(f) = oth(randi(rs, numel(oth)));
                end
                CL = [CL; repair(rs, ci, k)]; own = [own; i];
            end
        end
        fc = tl_evalw(L(CL), K); ne = ne + numel(fc);
        for i = 1:c.pop
            j = find(own == i); [bf, b] = max(fc(j));
            if bf > fit(i); fit(i) = bf; P(i, :) = CL(j(b), :); end
        end
        [fit, o] = sort(fit, 'descend'); P = P(o, :);
        jj = c.pop - c.dnew + 1:c.pop;
        for j = jj; P(j, :) = feasible(rs, F, nl, k); end
        fit(jj) = tl_evalw(L(P(jj, :)), K); ne = ne + numel(jj);
        hist(g, :) = [g, max(fit), mean(fit), ne];
    end
    [fb, ib] = max(fit); w = L(P(ib, :));
    if ik == 1
        nzf = find(w > 0); W0 = repmat(w, numel(nzf), 1);
        W0(sub2ind(size(W0), 1:numel(nzf), nzf)) = 0;
        loo = -inf(1, F); loo(nzf) = fb - tl_evalw(W0, K)';
        S.loo = loo;
    end
    S.w(ik, :) = w; S.f(ik) = fb; S.hist{ik} = hist;
end
ok = S.f >= S.f(1) - c.tol;
S.k = min(kl(ok)); S.wsel = S.w(kl == S.k, :);
end

function ci = repair(rs, ci, k)
nz = find(ci > 1);
if numel(nz) > k; ci(nz(randperm(rs, numel(nz), numel(nz) - k))) = 1; end
end

function ci = feasible(rs, F, nl, k)
ci = ones(1, F); pos = randperm(rs, F, k); ci(pos) = randi(rs, [2 nl], 1, k);
end

function ci = project(ci, k, loo)
nz = find(ci > 1);
if numel(nz) <= k; return; end
M = sortrows([-ci(nz)', -loo(nz)', nz'], [1 2 3]);
ci(setdiff(nz, M(1:k, 3))) = 1;
end

function li = lev(w, L)
[~, li] = ismember(w, L);
end
