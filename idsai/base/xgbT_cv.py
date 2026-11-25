import json
import os
import sys

import numpy as np
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.chdir(config.WORK)
s = np.load('split.npz', allow_pickle=True)
X = np.vstack([s['Xtr'], s['Xte']])
y = np.concatenate([s['ytr'], s['yte']])
fold = np.repeat(np.arange(10), len(X) // 10)
acc = []
for f in range(10):
    tr, te = fold != f, fold == f
    m = XGBClassifier(eval_metric='mlogloss', n_jobs=2, colsample_bytree=0.75, gamma=1.5, max_depth=6, min_child_weight=1,
                      subsample=1, learning_rate=0.3, n_estimators=100, tree_method='exact', random_state=0)
    m.fit(X[tr], y[tr])
    acc.append(float((m.predict(X[te]) == y[te]).mean()))
json.dump({'folds': acc}, open(os.path.join(config.out('base'), 'xgbT_cv10.json'), 'w'))
