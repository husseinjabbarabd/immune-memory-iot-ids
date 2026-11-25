import json
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from ais import layers
from zd_nested import names, dmin_same_sig, remap_y, tune

os.chdir(config.WORK)
K = 11
s = np.load('split.npz', allow_pickle=True)
Xtr, ytr, Xte, yte = s['Xtr'], s['ytr'], s['Xte'], s['yte']
classes = [str(c) for c in s['classes']]
N = classes.index('Normal')
T = [0.005, 0.01, 0.02, 0.05]
DG = np.r_[np.geomspace(1e-4, 10, 200), np.inf]
GRID = [(d, lr, ne) for d in (6, 10) for lr in (0.1, 0.3) for ne in (100, 300)]

rng = np.random.default_rng(21)
nidx = np.where(ytr == N)[0]
rng.shuffle(nidx)
cal = np.zeros(len(ytr), bool)
cal[nidx[320000:]] = True
Xc = Xtr[cal]

out = {'meta': dict(cal_rows=int(cal.sum()), fit_normal=int((ytr == N).sum() - cal.sum()), grid=GRID, targets=T)}


def xgb(d, lr, ne, nj):
    return XGBClassifier(eval_metric='mlogloss', n_jobs=nj, colsample_bytree=0.75, gamma=1.5,
                         max_depth=d, min_child_weight=1, subsample=1, learning_rate=lr, n_estimators=ne,
                         tree_method='exact', random_state=0)


def threads(hn, i):
    return 16 if hn == 'SYN group' and i > 0 else 2


SYN = [k for k in range(K) if 'SYN' in classes[k]]
rounds = [([h], classes[h]) for h in range(K) if h != N] + [(SYN, 'SYN group')]
for hs, hn in rounds:
    r = out.setdefault(hn, {})
    kc = [k for k in range(K) if k not in hs]
    Kc = len(kc)
    fit = ~np.isin(ytr, hs) & ~cal
    X = Xtr[fit]
    y = remap_y(ytr[fit], kc)
    nr = kc.index(N)
    ht = np.isin(yte, hs)
    nt = yte == N
    r['held_out'] = [classes[k] for k in hs]
    coef, w, cva, cvf = tune(X, y, Kc)

    def score(Xb):
        L = layers(X, y, Xb)
        S = sum(cf * L[n][:, :Kc] for cf, n in zip(coef, names) if cf) * w
        return np.array(kc)[S.argmax(1)], dmin_same_sig(X, Xb)

    pc, dc = score(Xc)
    pt, dt = score(Xte)
    attc, att = pc != N, pt != N
    cf_ = np.array([((dc > d) | attc).mean() for d in DG])
    im = dict(coef=[float(c) for c in coef], w=[float(v) for v in w], cv_acc=100 * cva, cv_f1m=100 * cvf,
              path_cal_fpr=100 * float(attc.mean()))
    for f in T:
        ok = np.where(cf_ <= f)[0]
        if attc.mean() > f or len(ok) == 0:
            im[str(f)] = dict(attainable=False)
            continue
        d = DG[ok.min()]
        eu = dt > d
        im[str(f)] = dict(attainable=True, delta=float(d), cal_fpr=100 * float(cf_[ok.min()]),
                          EUR=100 * float(eu[ht].mean()), ADR=100 * float((eu | att)[ht].mean()),
                          FPR=100 * float((eu | att)[nt].mean()),
                          per_class={classes[k]: [100 * float(eu[yte == k].mean()), 100 * float((eu | att)[yte == k].mean())]
                                     for k in hs})
    r['immune'] = im
    Xf, Xv, yf, yv = train_test_split(X, y, test_size=0.25, stratify=y, random_state=21)
    g = []
    for i, p in enumerate(GRID):
        pv = xgb(*p, threads(hn, i)).fit(Xf, yf).predict(Xv)
        g.append(accuracy_score(yv, pv) + 0.5 * f1_score(yv, pv, average='macro'))
    best = GRID[int(np.argmax(g))]
    m = xgb(*best, threads(hn, len(GRID))).fit(X, y)
    Pc = m.predict_proba(Xc)
    Pt = m.predict_proba(Xte)
    pc, mc = Pc.argmax(1), Pc.max(1)
    pt, mt = Pt.argmax(1), Pt.max(1)
    att = pt != nr
    u = float((pc != nr).mean())
    xr = dict(grid_obj=g, best=list(best), path_cal_fpr=100 * u, path_cal_rows=int((pc != nr).sum()))
    for f in T:
        need = f - u
        if need < 0:
            xr[str(f)] = dict(attainable=False)
            continue
        if need == 0:
            th = 0.0
        else:
            ms = np.sort(mc[pc == nr])
            k = int(round(need * len(mc)))
            th = float(ms[k - 1]) + 1e-12 if k > 0 else 0.0
        fl = mt < th
        xr[str(f)] = dict(attainable=True, th=th, cal_fpr=100 * float(((mc < th) | (pc != nr)).mean()),
                          EUR=100 * float(fl[ht].mean()), ADR=100 * float((fl | att)[ht].mean()),
                          FPR=100 * float((fl | att)[nt].mean()),
                          per_class={classes[k]: [100 * float(fl[yte == k].mean()), 100 * float((fl | att)[yte == k].mean())]
                                     for k in hs})
    r['xgb'] = xr
json.dump(out, open(os.path.join(config.out('nested'), 'nested_fpr.json'), 'w'), indent=1)
