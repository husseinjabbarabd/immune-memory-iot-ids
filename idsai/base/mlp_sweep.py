import json
import os
import sys

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.chdir(config.WORK)
s = np.load('split.npz', allow_pickle=True)
Xtr, ytr, Xte, yte = s['Xtr'], s['ytr'], s['Xte'], s['yte']
sc = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
res = []
for H in [2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32]:
    m = MLPClassifier(hidden_layer_sizes=(H,), activation='relu', solver='adam',
                      alpha=1e-4, batch_size=256, learning_rate_init=1e-3,
                      max_iter=60, early_stopping=True, n_iter_no_change=6,
                      validation_fraction=0.1, random_state=1)
    m.fit(Xtr_s, ytr)
    p = m.predict(Xte_s)
    nparam = sum(c.size for c in m.coefs_) + sum(b.size for b in m.intercepts_)
    res.append(dict(H=H, params=int(nparam), acc=accuracy_score(yte, p), f1w=f1_score(yte, p, average='weighted'),
                    f1m=f1_score(yte, p, average='macro'), iters=m.n_iter_))
json.dump(res, open(os.path.join(config.out('base'), 'mlp_sweep.json'), 'w'))
