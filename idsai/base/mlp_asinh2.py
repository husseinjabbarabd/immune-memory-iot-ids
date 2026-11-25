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
fn = os.path.join(config.out('base'), 'mlp_asinh.json')
s = np.load('split.npz', allow_pickle=True)
Xtr, ytr, Xte, yte = s['Xtr'].copy(), s['ytr'], s['Xte'].copy(), s['yte']
Xtr[:, 0] = np.arcsinh(1e6 * Xtr[:, 0])
Xte[:, 0] = np.arcsinh(1e6 * Xte[:, 0])
sc = StandardScaler().fit(Xtr)
Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
res = json.load(open(fn))
for H in [16, 32]:
    accs = []
    for sd in [0, 1]:
        m = MLPClassifier(hidden_layer_sizes=(H,), activation='relu', solver='adam',
                          alpha=1e-4, batch_size=256, learning_rate_init=1e-3, max_iter=100,
                          early_stopping=True, n_iter_no_change=8, validation_fraction=0.1,
                          random_state=sd).fit(Xtr, ytr)
        p = m.predict(Xte)
        accs.append((accuracy_score(yte, p), f1_score(yte, p, average='weighted'), f1_score(yte, p, average='macro')))
    a = np.array(accs)
    i = a[:, 0].argmax()
    res.append(dict(H=H, acc_best=float(a[i, 0]), f1w=float(a[i, 1]), f1m=float(a[i, 2]), acc_mean=float(a[:, 0].mean()),
                    acc_std=float(a[:, 0].std())))
json.dump(res, open(fn, 'w'))
