function [RES, ex] = tl_metrics(s, taus, pAll, y, ben, c, nC)
if nargin < 7; nC = max(y); cl = 1:nC; else; cl = find(accumarray(y(:), 1, [nC 1]) > 0)'; end
att = setdiff(cl, ben);
isA = y ~= ben; nA = sum(isA); nB = numel(y) - nA;
w = ones(numel(y), 1); w(~isA) = c.nb_ref / nB; w(isA) = c.na_ref / nA;
rk = tiedrank(s); auc = (sum(rk(isA)) - nA * (nA + 1) / 2) / (nA * nB);
op = {'OP-SPEC'; 'OP-F2'; 'OP-R95'}; R = zeros(3, 19);
for o = 1:3
    fw = s >= taus(o);
    b = binm(isA, fw, ones(size(y))); bw = binm(isA, fw, w);
    pr = pAll; pr(~fw) = ben;
    [fA, fM, ~, ~, FB] = am(cm(y(fw), pr(fw), ones(sum(fw), 1), nC), cl);
    [fAw, fMw] = am(cm(y(fw), pr(fw), w(fw), nC), cl);
    ra = fw & isA;
    [aA, aM, PA, RA, FA] = am(cm(y(ra), pr(ra), ones(sum(ra), 1), nC), att);
    [eA, eM] = am(cm(y, pr, ones(size(y)), nC), cl);
    eAw = am(cm(y, pr, w, nC), cl);
    R(o, :) = [taus(o), sum(fw), b.acc, b.mf1, b.rec, b.spec, b.f2, bw.acc, fA, fM, FB(ben), fAw, fMw, sum(ra), aA, aM, eA, eM, eAw];
    ex.pc{o} = [PA, RA, FA];
end
RES = array2table(R, 'VariableNames', {'tau', 'nFwd', 'P1_Acc', 'P1_MF1', 'P1_Rec', 'P1_Spec', 'P1_F2', 'P1_Acc_pc', ...
    'fwd_Acc', 'fwd_MF1', 'fwd_BenignF1', 'fwd_Acc_pc', 'fwd_MF1_pc', 'att_n', 'att_Acc', 'att_MF1', 'e2e_Acc', 'e2e_MF1', 'e2e_Acc_pc'});
RES = addvars(RES, op, 'Before', 1, 'NewVariableNames', 'opPoint');
pop = (~isA & s >= taus(1)) | (isA & s >= taus(3));
pr = pAll; pr(~pop) = ben;
[ex.popA, ex.popM] = am(cm(y(pop), pr(pop), w(pop), nC), cl);
pr = pAll; pr(s < taus(3)) = ben;
ex.cmR = cm(y, pr, ones(size(y)), nC);
ex.fa = sum(~isA & pr ~= ben) / numel(y);
ex.auc = auc;
end

function m = binm(P, Q, w)
tp = sum(w(P & Q)); fn = sum(w(P & ~Q)); tn = sum(w(~P & ~Q)); fp = sum(w(~P & Q));
m.acc = (tp + tn) / (tp + tn + fp + fn); m.rec = tp / (tp + fn); m.spec = tn / (tn + fp);
m.f2 = 5 * tp / (5 * tp + 4 * fn + fp);
m.mf1 = (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fn + fp)) / 2;
end

function M = cm(y, p, w, nC)
if isempty(y); M = zeros(nC); return; end
M = accumarray([y(:), p(:)], w(:), [nC nC]);
end

function [acc, mf1, P, R, F] = am(M, cls)
acc = trace(M) / max(eps, sum(M(:)));
nC = size(M, 1); P = zeros(nC, 1); R = zeros(nC, 1); F = zeros(nC, 1);
for k = 1:nC
    tp = M(k, k); cs = sum(M(:, k)); rs = sum(M(k, :));
    if cs > 0; P(k) = tp / cs; end
    if rs > 0; R(k) = tp / rs; end
    if P(k) + R(k) > 0; F(k) = 2 * P(k) * R(k) / (P(k) + R(k)); end
end
mf1 = mean(F(cls));
end
