import json
import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import ais

os.chdir(config.WORK)
s = np.load('split.npz', allow_pickle=True)
Xtr, ytr = s['Xtr'], s['ytr']
c = json.load(open(config.TUNED))
w = np.array(c['w'], np.float32)
base = [0.0025, 0.01, 0.02, 0.05]
cb = [0.1, 1.0, 1.0, 0.25]
cE, cS = 32.0, 0.0005
folds = list(StratifiedKFold(3).split(Xtr, ytr))


def run(hs):
    ais.HS = hs
    a = []
    f = []
    for tr, va in folds:
        L = ais.layers(Xtr[tr], ytr[tr], Xtr[va])
        S = cE * L['E'] + cS * L['SIG'] + sum(cc * L[f'B{h}'] for cc, h in zip(cb, hs))
        p = (S * w).argmax(1)
        a.append((p == ytr[va]).mean())
        f.append(f1_score(ytr[va], p, average='macro'))
    return float(np.mean(a)), float(np.mean(f))


cfg = [('all x%s' % k, [h * k for h in base]) for k in [0.5, 0.75, 1.0, 1.5, 2.0]]
for i, h in enumerate(base):
    for k in [0.5, 2.0]:
        hs = list(base)
        hs[i] = h * k
        cfg.append((f'h{i+1}={h} x{k}', hs))
out = []
for name, hs in cfg:
    a, f = run(hs)
    out.append(dict(config=name, hs=hs, cv_acc=a, cv_f1m=f))
json.dump(out, open(os.path.join(config.out('base'), 'sens.json'), 'w'), indent=1)
