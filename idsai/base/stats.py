import json
import os
import sys

import numpy as np
from scipy.stats import chi2, binomtest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.chdir(config.WORK)
s = np.load('split.npz', allow_pickle=True)
yte = s['yte']
pa = np.load('final_test_eval.npz')['pred']
px = np.load('xgb_test_pred.npy')
ca, cx = pa == yte, px == yte
b = int((ca & ~cx).sum())
c = int((~ca & cx).sum())
st = (abs(b - c) - 1) ** 2 / (b + c)
p = chi2.sf(st, 1)
pe = binomtest(b, b + c, 0.5).pvalue
d = ca.mean() - cx.mean()
n = len(yte)
se = np.sqrt(b + c - (b - c) ** 2 / n) / n
ci = (d - 1.96 * se, d + 1.96 * se)
rng = np.random.default_rng(1)
diff = ca.astype(np.int8) - cx.astype(np.int8)
bs = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(2000)])
bci = np.percentile(bs, [2.5, 97.5])
out = dict(ais_acc=float(ca.mean()), xgb_acc=float(cx.mean()), ais_only=b, xgb_only=c, mcnemar_chi2=float(st),
           mcnemar_p=float(p), exact_p=float(pe), diff_pp=float(100 * d), ci95_pp=[float(100 * ci[0]), float(100 * ci[1])],
           boot_ci95_pp=[float(100 * bci[0]), float(100 * bci[1])])
json.dump(out, open(os.path.join(config.out('base'), 'mcnemar.json'), 'w'), indent=1)
