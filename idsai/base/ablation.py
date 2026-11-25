import json
import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from ais import layers

os.chdir(config.WORK)
s = np.load('split.npz', allow_pickle=True)
Xtr, ytr, Xte, yte = s['Xtr'], s['ytr'], s['Xte'], s['yte']
c = json.load(open(config.TUNED))
w = np.array(c['w'], np.float32)
co = dict(zip(c['names'], c['coef']))
B = ['B0.0025', 'B0.01', 'B0.02', 'B0.05']
V = {'Signature layer only': ['SIG'], 'Exact layer only': ['E'], 'Timing bins only': B,
     'Exact + signature': ['E', 'SIG'], 'Timing bins + signature': B + ['SIG'], 'Exact + timing bins': ['E'] + B,
     'Full immune memory (E + bins + signature)': ['E'] + B + ['SIG']}


def score(L, ya, keys):
    S = sum(co[k] * L[k] for k in keys) * w
    prior = np.bincount(ya, minlength=11)
    z = S.sum(1) == 0
    p = S.argmax(1)
    p[z] = prior.argmax()
    return p


res = {k: {'cv_acc': [], 'cv_f1m': []} for k in V}
for tr, va in StratifiedKFold(3).split(Xtr, ytr):
    L = layers(Xtr[tr], ytr[tr], Xtr[va])
    for k, keys in V.items():
        p = score(L, ytr[tr], keys)
        res[k]['cv_acc'].append((p == ytr[va]).mean())
        res[k]['cv_f1m'].append(f1_score(ytr[va], p, average='macro'))
L = layers(Xtr, ytr, Xte)
for k, keys in V.items():
    p = score(L, ytr, keys)
    r = res[k]
    r['test_acc'] = float((p == yte).mean())
    r['test_f1m'] = float(f1_score(yte, p, average='macro'))
    r['cv_acc'] = float(np.mean(r['cv_acc']))
    r['cv_f1m'] = float(np.mean(r['cv_f1m']))
json.dump(res, open(os.path.join(config.out('base'), 'ablation.json'), 'w'), indent=1)
