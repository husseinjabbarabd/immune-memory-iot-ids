import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.chdir(config.WORK)
s = np.load('split.npz', allow_pickle=True)
yte = s['yte']
ite = s['ite']
pa = np.load('final_test_eval.npz')['pred']
pt = np.load('xgbT_test_pred.npy')
o = np.argsort(ite)
d = ((pa == yte).astype(np.int64) - (pt == yte).astype(np.int64))[o]
y = yte[o]
n = len(y)
R = 5000
res = {}
for L in [20, 200, 1000, 2000, 5000]:
    rng = np.random.default_rng(7)
    tot = np.zeros(R)
    for c in np.unique(y):
        dc = d[y == c]
        m = len(dc)
        cs = np.r_[0, np.cumsum(np.r_[dc, dc])]
        nb = -(-m // L)
        st = rng.integers(0, m, (R, nb))
        bs = cs[st + L] - cs[st]
        last = m - (nb - 1) * L
        bs[:, -1] = cs[st[:, -1] + last] - cs[st[:, -1]]
        tot += bs.sum(1)
    reps = 100 * tot / n
    res[L] = dict(ci95=[round(float(np.percentile(reps, 2.5)), 3), round(float(np.percentile(reps, 97.5)), 3)],
                  mean=round(float(reps.mean()), 4), share_le0=float((reps <= 0).mean()))
json.dump(dict(observed_diff_pp=100 * float(d.mean()), results=res),
          open(os.path.join(config.out('base'), 'circ_block.json'), 'w'), indent=1)
