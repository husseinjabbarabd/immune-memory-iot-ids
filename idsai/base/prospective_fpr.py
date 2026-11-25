import json
import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from ais import layers
from zd_nested import dmin_same_sig

os.chdir(config.WORK)
s = np.load('split.npz', allow_pickle=True)
Xtr, ytr, Xte, yte = s['Xtr'], s['ytr'], s['Xte'], s['yte']
classes = [str(c) for c in s['classes']]
K = 11
N = classes.index('Normal')
c = json.load(open(config.TUNED))
coef, names = c['coef'], c['names']
w = np.array(c['w'], np.float32)
T = [0.005, 0.01, 0.02, 0.05]
DG = np.r_[np.geomspace(1e-4, 10, 200), np.inf]
out = {}


def immune_pred(Xa, ya_full, h, Xb):
    keep = ya_full != h
    kc = [k for k in range(K) if k != h]
    m = np.full(K, -1)
    m[kc] = np.arange(len(kc))
    ya2 = m[ya_full[keep]]
    L = layers(Xa[keep], ya2, Xb)
    S = sum(cf * L[n][:, :len(kc)] for cf, n in zip(coef, names) if cf) * w[kc]
    return np.array(kc)[S.argmax(1)], dmin_same_sig(Xa[keep], Xb)


folds = list(StratifiedKFold(3).split(Xtr, ytr))
fpr_cv = {h: [] for h in range(K) if h != N}
for tr, va in folds:
    nm = ytr[va] == N
    for h in fpr_cv:
        p, dm = immune_pred(Xtr[tr], ytr[tr], h, Xtr[va][nm])
        att = p != N
        fpr_cv[h].append([float(((dm > d) | att).mean()) for d in DG])
res = {}
for h in fpr_cv:
    cv = np.mean(fpr_cv[h], 0)
    p, dm = immune_pred(Xtr, ytr, h, Xte)
    att = p != N
    ht = yte == h
    nt = yte == N
    r = {}
    for f in T:
        ok = np.where(cv <= f)[0]
        if len(ok) == 0:
            r[str(f)] = dict(attainable=False, cv_fpr_min=float(cv.min()))
            continue
        d = DG[ok.min()]
        eu = dm > d
        r[str(f)] = dict(attainable=True, delta=float(d), cv_fpr=float(cv[ok.min()]), EUR=100 * float(eu[ht].mean()),
                         ADR=100 * float((eu | att)[ht].mean()), FPR=100 * float((eu | att)[nt].mean()))
    res[classes[h]] = r
out['immune'] = res
X1, X2 = Xtr.copy(), Xte.copy()
for X in (X1, X2):
    X[:, 0] = np.arcsinh(1e6 * X[:, 0])
rng = np.random.default_rng(21)
nidx = np.where(ytr == N)[0]
rng.shuffle(nidx)
fi, ca = nidx[:320000], nidx[320000:]
sc = StandardScaler().fit(X1[fi])
res = {}
for name, mk, sub in [('IF', lambda: IsolationForest(n_estimators=200, max_samples=4096, random_state=21, n_jobs=2), fi),
                      ('OCSVM', lambda: OneClassSVM(kernel='rbf', gamma='scale', nu=0.02), fi[:20000])]:
    m = mk().fit(sc.transform(X1[sub]))
    scal = -m.score_samples(sc.transform(X1[ca]))
    ste = -m.score_samples(sc.transform(X2))
    r = {}
    for f in T:
        th = np.quantile(scal, 1 - f)
        fl = ste > th
        r[str(f)] = dict(FPR=100 * float(fl[yte == N].mean()),
                         ADR={classes[h]: 100 * float(fl[yte == h].mean()) for h in range(K) if h != N})
    res[name] = r
out['oneclass'] = res
out['xgb'] = {}
for h in [k for k in range(K) if k != N]:
    keep = ytr != h
    X, y = Xtr[keep], ytr[keep]
    kc = [k for k in range(K) if k != h]
    m_ = np.full(K, -1)
    m_[kc] = np.arange(10)
    yy = m_[y]
    Nn = m_[N]
    Xf, Xc, yf, yc = train_test_split(X, yy, test_size=0.2, stratify=yy, random_state=21)
    m = XGBClassifier(eval_metric='mlogloss', n_jobs=2, colsample_bytree=0.75, gamma=1.5, max_depth=6, min_child_weight=1,
                      subsample=1, learning_rate=0.3, n_estimators=100, tree_method='exact', random_state=0).fit(Xf, yf)
    Pc = m.predict_proba(Xc[yc == Nn])
    Pt = m.predict_proba(Xte)
    pc, mc = Pc.argmax(1), Pc.max(1)
    pt, mt = Pt.argmax(1), Pt.max(1)
    att = pt != Nn
    r = {}
    for f in T:
        need = f - (pc != Nn).mean()
        if need <= 0:
            th = 0.0
        else:
            ms = np.sort(mc[pc == Nn])
            k = int(round(need * len(mc)))
            th = float(ms[k - 1]) + 1e-12 if k > 0 else 0.0
        fl = mt < th
        r[str(f)] = dict(EUR=100 * float(fl[yte == h].mean()), ADR=100 * float((fl | att)[yte == h].mean()),
                         FPR=100 * float((fl | att)[yte == N].mean()), cal_fpr=float(((mc < th) | (pc != Nn)).mean()))
    out['xgb'][classes[h]] = r
json.dump(out, open(os.path.join(config.out('base'), 'prospective_fpr.json'), 'w'), indent=1)
