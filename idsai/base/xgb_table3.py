import json
import os
import sys

os.environ.pop('OMP_NUM_THREADS', None)

import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.chdir(config.WORK)
s = np.load('split.npz', allow_pickle=True)
Xtr, ytr, Xte, yte = s['Xtr'], s['ytr'], s['Xte'], s['yte']
pa = np.load('final_test_eval.npz')['pred']
m = XGBClassifier(learning_rate=0.1, max_depth=10, gamma=0, colsample_bytree=1, eval_metric='mlogloss', n_jobs=2,
                  random_state=0)
m.fit(Xtr, ytr)
p = m.predict(Xte)
np.save('xgb_table3_pred.npy', p)
a, b = pa == yte, p == yte
B = int((a & ~b).sum())
C = int((~a & b).sum())
x = (abs(B - C) - 1) ** 2 / (B + C)
out = dict(acc=100 * b.mean(), mF1=100 * f1_score(yte, p, average='macro'), wF1=100 * f1_score(yte, p, average='weighted'),
           b=B, c=C, chi2=x, p=float(chi2.sf(x, 1)))
json.dump(out, open(os.path.join(config.out('base'), 'xgb_table3.json'), 'w'), indent=1)
