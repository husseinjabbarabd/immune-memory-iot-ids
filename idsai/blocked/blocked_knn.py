import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.chdir(config.WORK)
d = np.load('idsai19.npz')
X = d['X']
le = LabelEncoder()
y = le.fit_transform(d['y'])
K = len(le.classes_)
tr = []
te = []
for k in range(K):
    i = np.flatnonzero(y == k)
    n = len(i) - int(round(0.2 * len(i)))
    tr.append(i[:n])
    te.append(i[n:])
tr = np.sort(np.concatenate(tr))
te = np.sort(np.concatenate(te))
assert np.array_equal(te, np.load('blocked_preds.npz')['te'])
Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
out = {}


def keys(A, cols):
    return pd.util.hash_pandas_object(pd.DataFrame(A[:, cols]), index=False).values


def maj(k, yy):
    df = pd.DataFrame({'k': k, 'y': yy})
    c = df.groupby(['k', 'y']).size().reset_index(name='n')
    c = c.sort_values(['k', 'n', 'y'], ascending=[True, False, True]).drop_duplicates('k')
    return dict(zip(c.k, c.y))


allc = list(range(19))
sig = list(range(1, 19))
me, ms = maj(keys(Xtr, allc), ytr), maj(keys(Xtr, sig), ytr)
prior = np.bincount(ytr).argmax()
pl = np.array([me.get(a, ms.get(b, prior)) for a, b in zip(keys(Xte, allc), keys(Xte, sig))])
out['lookup'] = dict(acc=100 * float((pl == yte).mean()), f1m=100 * float(f1_score(yte, pl, average='macro')))
A, B = Xtr.astype(float), Xte.astype(float)
for Z in (A, B):
    Z[:, 0] = np.arcsinh(1e6 * Z[:, 0])
sc = StandardScaler().fit(A)
pk = KNeighborsClassifier(51, weights='distance', algorithm='kd_tree', n_jobs=2).fit(sc.transform(A), ytr).predict(sc.transform(B))
out['knn'] = dict(k=51, acc=100 * float((pk == yte).mean()), f1m=100 * float(f1_score(yte, pk, average='macro')))
np.savez('blocked_knn.npz', te=te, knn=pk, lookup=pl)
json.dump(out, open(os.path.join(config.out('blocked'), 'blocked_knn.json'), 'w'), indent=1)
