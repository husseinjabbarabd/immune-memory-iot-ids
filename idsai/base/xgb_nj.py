import json
import os
import sys

import numpy as np
import xgboost
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.chdir(config.WORK)
s = np.load('split.npz', allow_pickle=True)
Xtr, ytr, Xte, yte = s['Xtr'], s['ytr'], s['Xte'], s['yte']
p2 = np.load('xgbT_test_pred.npy')
d = {}
for nj in (1, 4):
    m = XGBClassifier(eval_metric='mlogloss', n_jobs=nj, colsample_bytree=0.75, gamma=1.5, max_depth=6, min_child_weight=1,
                      subsample=1, learning_rate=0.3, n_estimators=100, tree_method='exact', random_state=0)
    m.fit(Xtr, ytr)
    p = m.predict(Xte)
    np.save('xgbT_test_pred_nj%d.npy' % nj, p)
    d[str(nj)] = dict(n_jobs=nj, xgboost=xgboost.__version__, correct=int((p == yte).sum()), n=int(len(yte)),
                      acc=100 * float((p == yte).mean()), differs_from_nj2=int((p != p2).sum()))
json.dump(d, open(os.path.join(config.out('base'), 'xgb_nj.json'), 'w'), indent=1)
