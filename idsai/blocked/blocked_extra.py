import json
import os
import sys

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, recall_score
from scipy.stats import chi2, binomtest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.chdir(config.WORK)
d = np.load('idsai19.npz')
X = d['X'].astype(np.float64)
le = LabelEncoder()
y = le.fit_transform(d['y'])
cl = [str(c) for c in le.classes_]
K = len(cl)
N = cl.index('Normal')
P = dict(np.load('blocked_preds.npz'))
te = P['te']
R2 = np.load('blocked_xgb_retuned.npz')
R3 = np.load('blocked_knn.npz')
assert np.array_equal(R2['te'], te) and np.array_equal(R3['te'], te)
P['xgb_retuned'] = R2['xgb_retuned']
P['knn'] = R3['knn']
P['lookup'] = R3['lookup']
M = ['immune', 'xgb_tuned', 'xgb_default', 'xgb_retuned', 'knn', 'lookup']
tr = np.setdiff1d(np.arange(len(y)), te)
for k in range(K):
    i = np.flatnonzero(y == k)
    n = len(i) - int(round(0.2 * len(i)))
    assert np.array_equal(np.sort(i[n:]), np.sort(te[y[te] == k]))


def rows(A):
    return np.ascontiguousarray(A).view(np.dtype((np.void, A.dtype.itemsize * A.shape[1]))).ravel()


yt = y[te]
out = {}
out['models'] = {m: dict(acc=100 * float((P[m] == yt).mean()), f1m=100 * float(f1_score(yt, P[m], average='macro')),
                         recall={cl[k]: 100 * float(v) for k, v in enumerate(recall_score(yt, P[m], average=None))})
                 for m in M}
a = yt == cl.index('ARP spoofing')
out['arp_row'] = {m: {cl[k]: int(v) for k, v in enumerate(np.bincount(P[m][a], minlength=K)) if v} for m in M}
fa = rows(X)
ft = fa[te]
ftr = fa[tr]
seen = np.isin(ft, ftr)
sig = np.isin(rows(X[te][:, 1:]), rows(X[tr][:, 1:]))
g = {'seen': seen, 'known_sig_new_dt': ~seen & sig, 'unseen_sig': ~sig}
u, inv = np.unique(fa, return_inverse=True)
C = np.zeros((len(u), K), np.int64)
np.add.at(C, (inv, y), 1)
conf = (C > 0).sum(1)[inv[te]] > 1
Ct = np.zeros((len(u), K), np.int64)
np.add.at(Ct, (inv[tr], y[tr]), 1)
maj = Ct.argmax(1)[inv[te]]
for nm, s in g.items():
    yy = yt[s]
    vv = inv[te][s]
    uu, ii = np.unique(vv, return_inverse=True)
    Cg = np.zeros((len(uu), K), np.int64)
    np.add.at(Cg, (ii, yy), 1)
    forced = int(len(yy) - Cg.max(1).sum())
    r = dict(n=int(s.sum()), share=100 * float(s.mean()), conflicted=100 * float(conf[s].mean()) if s.any() else None,
             ceiling=100 * float(Cg.max(1).sum() / len(yy)) if s.any() else None, forced=forced)
    for m in M:
        e = int((P[m][s] != yy).sum())
        r[m] = dict(acc=100 * float((P[m][s] == yy).mean()), errors=e, avoidable=e - forced)
    if nm == 'seen':
        r['train_majority_differs'] = int((maj[s] != yy).sum())
        for m in M:
            r[m]['errors_equal_train_majority'] = int(((P[m][s] != yy) & (P[m][s] == maj[s])).sum())
    if nm == 'unseen_sig':
        r['true_normal_share'] = 100 * float((yy == N).mean())
        r['true_class_counts'] = {cl[k]: int(v) for k, v in enumerate(np.bincount(yy, minlength=K)) if v}
    out[nm] = r
S = np.load('split.npz', allow_pickle=True)
out['blocked_test_in_published_train'] = dict(n=int(np.isin(te, S['itr']).sum()), share=100 * float(np.isin(te, S['itr']).mean()))
uu, ii = np.unique(inv[te], return_inverse=True)
Ca = np.zeros((len(uu), K), np.int64)
np.add.at(Ca, (ii, yt), 1)
out['all'] = dict(ceiling=100 * float(Ca.max(1).sum() / len(yt)), forced=int(len(yt) - Ca.max(1).sum()),
                  **{m: dict(errors=int((P[m] != yt).sum())) for m in M})
o = np.argsort(te)
yo = yt[o]
n = len(yo)
R = 5000
for ref, key in (('xgb_tuned', 'block_bootstrap'), ('xgb_retuned', 'block_bootstrap_retuned'),
                 ('knn', 'block_bootstrap_knn'), ('lookup', 'block_bootstrap_lookup')):
    dd = ((P['immune'] == yt).astype(np.int64) - (P[ref] == yt).astype(np.int64))[o]
    bs_res = {}
    for L in [20, 200, 1000, 2000, 5000]:
        rng = np.random.default_rng(7)
        tot = np.zeros(R)
        for c in np.unique(yo):
            dc = dd[yo == c]
            m = len(dc)
            cs = np.r_[0, np.cumsum(np.r_[dc, dc])]
            nb = -(-m // L)
            st = rng.integers(0, m, (R, nb))
            bs = cs[st + L] - cs[st]
            last = m - (nb - 1) * L
            bs[:, -1] = cs[st[:, -1] + last] - cs[st[:, -1]]
            tot += bs.sum(1)
        reps = 100 * tot / n
        bs_res[L] = dict(ci95=[round(float(np.percentile(reps, 2.5)), 3), round(float(np.percentile(reps, 97.5)), 3)],
                         mean=round(float(reps.mean()), 4), share_le0=float((reps <= 0).mean()))
    a = P['immune'] == yt
    b = P[ref] == yt
    n10 = int((a & ~b).sum())
    n01 = int((~a & b).sum())
    x2 = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)
    out[key] = dict(observed_diff_pp=100 * float(dd.mean()), results=bs_res,
                    mcnemar=dict(immune_only=n10, xgb_only=n01, chi2_cc=x2, p_cc=float(chi2.sf(x2, 1)),
                                 p_exact=float(binomtest(n10, n10 + n01, 0.5).pvalue)))
json.dump(out, open(os.path.join(config.out('blocked'), 'blocked_extra.json'), 'w'), indent=1)
