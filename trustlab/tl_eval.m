function [RES, ex, taus, nNew] = tl_eval(sV, sT, B, P, c)
taus = tl_tau(sV, B.y(B.va) ~= B.ben, c);
fw = sT >= min(taus); need = find(fw & ~P.has); nNew = numel(need);
p = P.p;
if nNew > 0
    p(need) = tl_p2(B.cTr(:, P.f), B.y(B.tr), B.cTe(need, P.f), P.w(P.f), P.wexp(P.f), B.trig, numel(B.names), c);
end
pAll = B.ben * ones(numel(B.te), 1); pAll(fw) = p(fw);
[RES, ex] = tl_metrics(sT, taus, pAll, B.y(B.te), B.ben, c);
end
