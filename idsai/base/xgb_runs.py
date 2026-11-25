import json
import os
import sys

import numpy as np
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.chdir(config.WORK)
s = np.load('split.npz', allow_pickle=True)
Xtr, ytr, Xte, yte = s['Xtr'], s['ytr'], s['Xte'], s['yte']
m = XGBClassifier(eval_metric='mlogloss', n_jobs=2)
m.fit(Xtr, ytr)
np.save('xgb_test_pred.npy', m.predict(Xte))
X = np.vstack([Xtr, Xte])
y = np.concatenate([ytr, yte])
N = len(X)
fold = np.repeat(np.arange(10), N // 10)
acc = []
for f in range(10):
    tr, te = fold != f, fold == f
    m = XGBClassifier(eval_metric='mlogloss', n_jobs=2)
    m.fit(X[tr], y[tr])
    acc.append(float((m.predict(X[te]) == y[te]).mean()))
json.dump({'folds': acc}, open(os.path.join(config.out('base'), 'xgb_cv10.json'), 'w'))
