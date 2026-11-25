import json
import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2, binomtest
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from ais import layers, HS

os.chdir(config.WORK)
OUT = os.path.join(config.out('blocked'), 'idsai_blocked.json')
d = np.load('idsai19.npz')
X = d['X'].astype(np.float64)
ys = d['y']
le = LabelEncoder()
y = le.fit_transform(ys)
classes = [str(c) for c in le.classes_]
K = len(classes)
tr = []
te = []
for k in range(K):
    i = np.flatnonzero(y == k)
    n = len(i) - int(round(0.2 * len(i)))
    tr.append(i[:n])
    te.append(i[n:])
tr = np.sort(np.concatenate(tr))
te = np.sort(np.concatenate(te))
Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
out = dict(n_train=int(len(tr)), n_test=int(len(te)), test_per_class=np.bincount(yte, minlength=K).tolist(), classes=classes)
folds = list(StratifiedKFold(3).split(Xtr, ytr))
caches = []
yv_ = []
for a, b in folds:
    caches.append(layers(Xtr[a], ytr[a], Xtr[b]))
    yv_.append(ytr[b])
names = ['E'] + [f'B{h}' for h in HS] + ['SIG']


def evalc(coef, w):
    A = []
    F = []
    for C, yv in zip(caches, yv_):
        p = (sum(c * C[n] for c, n in zip(coef, names) if c) * w).argmax(1)
        A.append(accuracy_score(yv, p))
        F.append(f1_score(yv, p, average='macro'))
    return np.mean(A), np.mean(F)


coef = [10, 0, 0, 1, 0, 0, 0.001]
w = np.ones(K)
ba, bf = evalc(coef, w)
obj = ba + 0.5 * bf
gridc = [0, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32]
for it in range(3):
    for i in range(len(coef)):
        if names[i] == 'B0.01':
            continue
        best = coef[i]
        for g in (gridc if names[i] != 'SIG' else [0, 0.0005, 0.001, 0.005, 0.02, 0.1, 0.5]):
            c2 = list(coef)
            c2[i] = g
            a, f = evalc(c2, w)
            o = a + 0.5 * f
            if o > obj + 1e-6:
                obj, best, ba, bf = o, g, a, f
        coef[i] = best
gridw = [0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4]
for it in range(3):
    for c in range(K):
        best = w[c]
        for g in gridw:
            w2 = w.copy()
            w2[c] = g
            a, f = evalc(coef, w2)
            o = a + 0.5 * f
            if o > obj + 1e-6:
                obj, best, ba, bf = o, g, a, f
        w[c] = best
del caches
w32 = np.array(w, np.float32)
L = layers(Xtr, ytr, Xte)
P = {'immune': (sum(cf * L[n] for cf, n in zip(coef, names) if cf) * w32).argmax(1)}
out['immune_cfg'] = dict(coef=[float(c) for c in coef], w=[float(v) for v in w], cv_acc=100 * ba, cv_f1m=100 * bf)
for nm, kw in [('xgb_tuned', dict(colsample_bytree=0.75, gamma=1.5, max_depth=6, min_child_weight=1, subsample=1,
                                  learning_rate=0.3, n_estimators=100, tree_method='exact')),
               ('xgb_default', dict())]:
    P[nm] = XGBClassifier(eval_metric='mlogloss', n_jobs=2, random_state=0, **kw).fit(Xtr, ytr).predict(Xte)


def rows(A):
    return np.ascontiguousarray(A).view(np.dtype((np.void, A.dtype.itemsize * A.shape[1]))).ravel()


ft = rows(Xte)
seen = np.isin(ft, rows(Xtr))
sig = np.isin(rows(Xte[:, 1:]), rows(Xtr[:, 1:]))
u, inv = np.unique(ft, return_inverse=True)
C = np.zeros((len(u), K), np.int64)
np.add.at(C, (inv, yte), 1)
out['ceiling_test_rows'] = float(C.max(1).sum() / len(yte))
out['share_seen_vector'] = float(seen.mean())
out['share_known_signature_new_dt'] = float((~seen & sig).mean())
out['share_unseen_signature'] = float((~sig).mean())
for nm, p in P.items():
    out[nm] = dict(acc=100 * float((p == yte).mean()), f1m=100 * float(f1_score(yte, p, average='macro')),
                   recall={classes[k]: 100 * float(v) for k, v in enumerate(recall_score(yte, p, average=None))},
                   acc_seen=100 * float((p == yte)[seen].mean()), acc_unseen=100 * float((p == yte)[~seen].mean()))
a = P['immune'] == yte
b = P['xgb_tuned'] == yte
n10 = int((a & ~b).sum())
n01 = int((~a & b).sum())
out['mcnemar_immune_vs_xgb_tuned'] = dict(immune_only=n10, xgb_only=n01,
                                           chi2_cc=(abs(n10 - n01) - 1) ** 2 / (n10 + n01) if n10 + n01 else None,
                                           p_cc=float(chi2.sf((abs(n10 - n01) - 1) ** 2 / (n10 + n01), 1)) if n10 + n01 else None,
                                           p_exact=float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else None)
np.savez('blocked_preds.npz', te=te, **P)
json.dump(out, open(OUT, 'w'), indent=1)
