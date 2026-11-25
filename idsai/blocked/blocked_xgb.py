import json
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.chdir(config.WORK)
GRID = [(d, lr, ne) for d in (6, 10) for lr in (0.1, 0.3) for ne in (100, 300)]
d = np.load('idsai19.npz')
X = d['X'].astype(np.float64)
le = LabelEncoder()
y = le.fit_transform(d['y'])
cl = [str(c) for c in le.classes_]
K = len(cl)
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


def xgb(dp, lr, ne):
    return XGBClassifier(eval_metric='mlogloss', n_jobs=2, colsample_bytree=0.75, gamma=1.5, max_depth=dp,
                         min_child_weight=1, subsample=1, learning_rate=lr, n_estimators=ne, tree_method='exact',
                         random_state=0)


g = []
Xf, Xv, yf, yv = train_test_split(Xtr, ytr, test_size=0.25, stratify=ytr, random_state=21)
for p in GRID:
    pv = xgb(*p).fit(Xf, yf).predict(Xv)
    g.append(accuracy_score(yv, pv) + 0.5 * f1_score(yv, pv, average='macro'))
best = GRID[int(np.argmax(g))]
p = xgb(*best).fit(Xtr, ytr).predict(Xte)
np.savez('blocked_xgb_retuned.npz', te=te, xgb_retuned=p)
out = dict(grid=GRID, grid_obj=g, best=list(best), refit_rows=int(len(tr)),
           test=dict(acc=100 * float((p == yte).mean()), f1m=100 * float(f1_score(yte, p, average='macro')),
                     recall={cl[k]: 100 * float(v) for k, v in enumerate(recall_score(yte, p, average=None))}))
json.dump(out, open(os.path.join(config.out('blocked'), 'blocked_xgb.json'), 'w'), indent=1)
