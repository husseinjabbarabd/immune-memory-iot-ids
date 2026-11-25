import json
import os
import sys

import numpy as np
from scipy.stats import chi2, binomtest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.chdir(config.WORK)
s = np.load('split.npz', allow_pickle=True)
y = s['yte']
pa = np.load('final_test_eval.npz')['pred']
out = {}
for name, f in [('xgb_tuned', 'xgbT_test_pred.npy'), ('knn_k51', 'knn_test_pred.npy'),
                ('knn_k201_extension', 'knn2_test_pred.npy'), ('lookup', 'lookup_test_pred.npy')]:
    p = np.load(f)
    a, bb = pa == y, p == y
    b = int((a & ~bb).sum())
    c = int((~a & bb).sum())
    x = (abs(b - c) - 1) ** 2 / (b + c)
    out[name] = dict(b_only_immune=b, c_only_baseline=c, chi2_cc=round(x, 2), p=float(chi2.sf(x, 1)),
                     exact_p=float(binomtest(b, b + c, 0.5).pvalue), baseline_acc=100 * float(bb.mean()),
                     diff_pp=100 * float(a.mean() - bb.mean()))
json.dump(out, open(os.path.join(config.out('base'), 'mcnemar_baselines.json'), 'w'), indent=1)
