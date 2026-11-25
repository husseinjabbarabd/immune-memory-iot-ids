import os
import sys

import numpy as np
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.chdir(config.WORK)
s = np.load('split.npz', allow_pickle=True)
Xtr, ytr, Xte, yte = s['Xtr'], s['ytr'], s['Xte'], s['yte']
m = XGBClassifier(eval_metric='mlogloss', n_jobs=2, colsample_bytree=0.75, gamma=1.5, max_depth=6, min_child_weight=1,
                  subsample=1, learning_rate=0.3, n_estimators=100, tree_method='exact', random_state=0)
m.fit(Xtr, ytr)
np.save('xgbT_test_pred.npy', m.predict(Xte))
