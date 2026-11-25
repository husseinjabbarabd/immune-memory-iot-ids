import json
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from ais import layers

os.chdir(config.WORK)
s = np.load('split.npz', allow_pickle=True)
X = np.vstack([s['Xtr'], s['Xte']])
y = np.concatenate([s['ytr'], s['yte']])
c = json.load(open(config.TUNED))
w = np.array(c['w'], np.float32)
N = len(X)
fold = np.repeat(np.arange(10), N // 10)
accs = []
for f in range(10):
    tr, te = fold != f, fold == f
    L = layers(X[tr], y[tr], X[te])
    S = sum(cf * L[n] for cf, n in zip(c['coef'], c['names']) if cf) * w
    accs.append(accuracy_score(y[te], S.argmax(1)))
json.dump({'cv10_mean': float(np.mean(accs)), 'cv10_std': float(np.std(accs)), 'folds': accs},
          open(os.path.join(config.out('base'), 'cv10.json'), 'w'))
