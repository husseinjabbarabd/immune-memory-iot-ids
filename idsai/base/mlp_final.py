import json
import os
import sys

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, cohen_kappa_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.chdir(config.WORK)
s = np.load('split.npz', allow_pickle=True)
Xtr, ytr, Xte, yte, classes = s['Xtr'], s['ytr'], s['Xte'], s['yte'], [str(c) for c in s['classes']]
sc = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
stab = {}
for H in [4, 5]:
    accs = []
    for sd in range(5):
        m = MLPClassifier(hidden_layer_sizes=(H,), activation='relu', solver='adam',
                          alpha=1e-4, batch_size=256, learning_rate_init=1e-3, max_iter=120,
                          early_stopping=True, n_iter_no_change=10, validation_fraction=0.1, random_state=sd)
        m.fit(Xtr_s, ytr)
        accs.append(accuracy_score(yte, m.predict(Xte_s)))
    stab[str(H)] = dict(mean=float(np.mean(accs)), std=float(np.std(accs)), min=float(min(accs)), max=float(max(accs)))
best = None
for sd in range(5):
    m = MLPClassifier(hidden_layer_sizes=(5,), activation='relu', solver='adam', alpha=1e-4,
                      batch_size=256, learning_rate_init=1e-3, max_iter=200, early_stopping=True,
                      n_iter_no_change=12, validation_fraction=0.1, random_state=sd).fit(Xtr_s, ytr)
    v = m.best_validation_score_
    if best is None or v > best[0]:
        best = (v, m, sd)
m = best[1]
p = m.predict(Xte_s)
nparam = sum(c.size for c in m.coefs_) + sum(b.size for b in m.intercepts_)
a = accuracy_score(yte, p)
fw = f1_score(yte, p, average='weighted')
fm = f1_score(yte, p, average='macro')
json.dump(dict(H=5, params=int(nparam), acc=a, f1w=fw, f1m=fm, seed=best[2], kappa=float(cohen_kappa_score(yte, p)),
               classes=classes, recall=recall_score(yte, p, average=None).tolist(),
               f1=f1_score(yte, p, average=None).tolist(), seed_stability=stab),
          open(os.path.join(config.out('base'), 'mlp_final.json'), 'w'))
np.savez('mlp_final.npz', cm=confusion_matrix(yte, p), pred=p)
