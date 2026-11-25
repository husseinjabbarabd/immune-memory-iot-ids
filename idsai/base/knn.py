import json
import os
import sys

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, cohen_kappa_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.chdir(config.WORK)
s = np.load('split.npz', allow_pickle=True)
Xtr, ytr, Xte, yte = s['Xtr'].astype(float), s['ytr'], s['Xte'].astype(float), s['yte']
for X in (Xtr, Xte):
    X[:, 0] = np.arcsinh(1e6 * X[:, 0])
out = {}
Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=21)
sc = StandardScaler().fit(Xa)
A, V = sc.transform(Xa), sc.transform(Xv)
best = None
for k in [1, 5, 15, 51]:
    m = KNeighborsClassifier(k, weights='distance' if k > 1 else 'uniform', algorithm='kd_tree', n_jobs=2).fit(A, ya)
    out[f'val_k{k}'] = 100 * (m.predict(V) == yv).mean()
    if best is None or out[f'val_k{k}'] > out[f'val_k{best}']:
        best = k
sc = StandardScaler().fit(Xtr)
m = KNeighborsClassifier(best, weights='distance' if best > 1 else 'uniform', algorithm='kd_tree', n_jobs=2).fit(
    sc.transform(Xtr), ytr)
p = m.predict(sc.transform(Xte))
np.save('knn_test_pred.npy', p)
out.update(k=best, acc=100 * (p == yte).mean(), wF1=100 * f1_score(yte, p, average='weighted'),
           mF1=100 * f1_score(yte, p, average='macro'), kappa=cohen_kappa_score(yte, p))
json.dump(out, open(os.path.join(config.out('base'), 'knn.json'), 'w'), indent=1)
