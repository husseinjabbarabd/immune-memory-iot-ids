import json
import os
import sys

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.chdir(config.WORK)
d = np.load('idsai19.npz')
le = LabelEncoder()
y = le.fit_transform(d['y'])
cl = [str(c) for c in le.classes_]
P = dict(np.load('blocked_preds.npz'))
te = P['te']
yt = y[te]
P['xgb_retuned'] = np.load('blocked_xgb_retuned.npz')['xgb_retuned']
R = np.load('blocked_knn.npz')
P['knn'] = R['knn']
P['lookup'] = R['lookup']
M = ['immune', 'xgb_retuned', 'xgb_tuned', 'xgb_default', 'knn', 'lookup']
out = {'f1': {}, 'precision': {}, 'lead_by_class': {}}
for m in M:
    out['f1'][m] = {cl[k]: 100 * float(v) for k, v in enumerate(f1_score(yt, P[m], average=None))}
    out['precision'][m] = {cl[k]: 100 * float(v) for k, v in enumerate(precision_score(yt, P[m], average=None, zero_division=0))}
for ref in M[1:]:
    a = P['immune'] == yt
    b = P[ref] == yt
    out['lead_by_class'][ref] = {cl[k]: dict(immune_only=int((a & ~b & (yt == k)).sum()), ref_only=int((~a & b & (yt == k)).sum()))
                                 for k in range(len(cl))}
json.dump(out, open(os.path.join(config.out('blocked'), 'blocked_perclass.json'), 'w'), indent=1)
