import json
import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from ais import layers
from zd_nested import dmin_same_sig, DS

os.chdir(config.WORK)
s = np.load('split.npz', allow_pickle=True)
Xtr, ytr = s['Xtr'], s['ytr']
classes = [str(c) for c in s['classes']]
K = 11
NORMAL = classes.index('Normal')
c = json.load(open(config.TUNED))
w = np.array(c['w'], np.float32)
coef, names = c['coef'], c['names']
folds = list(StratifiedKFold(3).split(Xtr, ytr))
cache = []
for tr, va in folds:
    Xa, ya, Xb, yb = Xtr[tr], ytr[tr], Xtr[va], ytr[va]
    for ci in range(K):
        if ci == NORMAL:
            continue
        keep = ya != ci
        cls_keep = [k for k in range(K) if k != ci]
        remap = {k: i for i, k in enumerate(cls_keep)}
        ya2 = np.array([remap[v] for v in ya[keep]])
        L = layers(Xa[keep], ya2, Xb)
        S = sum(cf * L[n][:, :len(cls_keep)] for cf, n in zip(coef, names) if cf) * w[cls_keep]
        pred = np.array(cls_keep)[S.argmax(1)]
        dm = dmin_same_sig(Xa[keep], Xb)
        cache.append(dict(ci=ci, zd=yb == ci, nm=yb == NORMAL, pred=pred, dmin=dm))
results = []
for d in DS:
    ADR = {}
    EUR = {}
    FPR = []
    for e in cache:
        eu = e['dmin'] > d
        al = eu | (e['pred'] != NORMAL)
        ADR.setdefault(e['ci'], []).append(al[e['zd']].mean())
        EUR.setdefault(e['ci'], []).append(eu[e['zd']].mean())
        FPR.append(al[e['nm']].mean())
    a = np.mean([np.mean(v) for v in ADR.values()]) * 100
    eu_ = np.mean([np.mean(v) for v in EUR.values()]) * 100
    f = np.mean(FPR) * 100
    results.append((d, a, eu_, f))
feas = [r for r in results if r[3] <= 2.5]
d, a, eu_, f = max(feas, key=lambda r: r[1])
config.out('base')
json.dump({'d': float(d), 'cv_ADR': a, 'cv_EUR': eu_, 'cv_FPR': f}, open(config.RULE, 'w'))
