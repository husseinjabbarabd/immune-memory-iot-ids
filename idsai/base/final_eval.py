import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from ais import layers
from metrics import report

os.chdir(config.WORK)
s = np.load('split.npz', allow_pickle=True)
Xtr, ytr, Xte, yte, classes = s['Xtr'], s['ytr'], s['Xte'], s['yte'], s['classes']
c = json.load(open(config.TUNED))
w = np.array(c['w'], np.float32)
L = layers(Xtr, ytr, Xte)
S = sum(cf * L[n] for cf, n in zip(c['coef'], c['names']) if cf) * w
p = S.argmax(1)
r = report(yte, p)
Pn = S / np.clip(S.sum(1, keepdims=True), 1e-12, None)
auc = roc_auc_score(yte, Pn, multi_class='ovo', average='weighted')
json.dump(dict(c, test_acc=r['acc'], test_f1m=r['f1m'], test_f1w=r['f1w'], test_pw=r['pw'], test_pm=r['pm'],
               test_kappa=r['kappa'], test_roc_auc_ovo_weighted=float(auc), classes=[str(k) for k in classes],
               test_recall=r['recall'], test_f1=r['f1']),
          open(os.path.join(config.out('base'), 'final_results.json'), 'w'))
np.savez('final_test_eval.npz', pred=p, cm=confusion_matrix(yte, p))
