import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.chdir(config.WORK)
s = np.load('split.npz', allow_pickle=True)
Xtr, ytr, Xte, yte = s['Xtr'], s['ytr'], s['Xte'], s['yte']


def keys(X, cols):
    return pd.util.hash_pandas_object(pd.DataFrame(X[:, cols]), index=False).values


allc = list(range(19))
sig = list(range(1, 19))


def maj(ktr, y):
    df = pd.DataFrame({'k': ktr, 'y': y})
    c = df.groupby(['k', 'y']).size().reset_index(name='n')
    c = c.sort_values(['k', 'n', 'y'], ascending=[True, False, True]).drop_duplicates('k')
    return dict(zip(c.k, c.y))


def run(Xa, ya, Xb):
    me, ms = maj(keys(Xa, allc), ya), maj(keys(Xa, sig), ya)
    ke, ks = keys(Xb, allc), keys(Xb, sig)
    prior = np.bincount(ya).argmax()
    return np.array([me.get(a, ms.get(b, prior)) for a, b in zip(ke, ks)])


p = run(Xtr, ytr, Xte)
out = dict(acc=100 * (p == yte).mean(), wF1=100 * f1_score(yte, p, average='weighted'),
           mF1=100 * f1_score(yte, p, average='macro'), kappa=cohen_kappa_score(yte, p))
cv = []
for tr, va in StratifiedKFold(3, shuffle=True, random_state=21).split(Xtr, ytr):
    q = run(Xtr[tr], ytr[tr], Xtr[va])
    cv.append(100 * (q == ytr[va]).mean())
out['cv3_acc'] = cv
np.save('lookup_test_pred.npy', p)
json.dump(out, open(os.path.join(config.out('base'), 'lookup.json'), 'w'), indent=1)
