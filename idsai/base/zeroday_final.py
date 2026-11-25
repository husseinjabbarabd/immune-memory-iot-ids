import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from ais import layers
from zd_nested import dmin_same_sig

os.chdir(config.WORK)
s = np.load('split.npz', allow_pickle=True)
Xtr, ytr, Xte, yte = s['Xtr'], s['ytr'], s['Xte'], s['yte']
classes = [str(c) for c in s['classes']]
K = 11
NORMAL = classes.index('Normal')
c = json.load(open(config.TUNED))
w = np.array(c['w'], np.float32)
coef, names = c['coef'], c['names']
D = json.load(open(config.RULE))['d']
rows = []
for ci in range(K):
    if ci == NORMAL:
        continue
    keep = ytr != ci
    cls_keep = [k for k in range(K) if k != ci]
    remap = {k: i for i, k in enumerate(cls_keep)}
    ytr2 = np.array([remap[v] for v in ytr[keep]])
    L = layers(Xtr[keep], ytr2, Xte)
    S = sum(cf * L[n][:, :len(cls_keep)] for cf, n in zip(coef, names) if cf) * w[cls_keep]
    pred = np.array(cls_keep)[S.argmax(1)]
    dm = dmin_same_sig(Xtr[keep], Xte)
    eu = dm > D
    state = np.where(eu, 0, np.where(pred != NORMAL, 1, 2))
    m = yte == ci
    n = int(m.sum())
    EU, KA, NM = [int((state[m] == v).sum()) for v in (0, 1, 2)]
    assert EU + KA + NM == n
    EUR, KAR, NR = 100 * EU / n, 100 * KA / n, 100 * NM / n
    ADR = EUR + KAR
    assert abs(EUR + KAR + NR - 100) < 1e-9 and abs(ADR - (100 - NR)) < 1e-9
    fpr = 100 * (state[yte == NORMAL] != 2).mean()
    rows.append(dict(attack=classes[ci], N=n, EUR=EUR, KAR=KAR, NR=NR, ADR=ADR, FPR=fpr))
json.dump(rows, open(os.path.join(config.out('base'), 'zeroday_final_results.json'), 'w'))
