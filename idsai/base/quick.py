import json
import os
import sys

import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.chdir(config.WORK)
s = np.load('split.npz', allow_pickle=True)
yte = s['yte']
ite = s['ite']
pa = np.load('final_test_eval.npz')['pred']
pt = np.load('xgbT_test_pred.npy')
out = {}
for n, p in [('AIS', pa), ('XGB_tuned', pt)]:
    out[n] = dict(acc=100 * accuracy_score(yte, p), wF1=100 * f1_score(yte, p, average='weighted'),
                  mF1=100 * f1_score(yte, p, average='macro'), kappa=cohen_kappa_score(yte, p))
o = np.argsort(ite)
d = ((pa == yte).astype(int) - (pt == yte).astype(int))[o]
io = ite[o]
rng = np.random.default_rng(7)
bb = {}
for B in [100, 1000, 10000, 50000]:
    blk = io // B
    ub, inv = np.unique(blk, return_inverse=True)
    S = np.bincount(inv, weights=d)
    C = np.bincount(inv)
    k = len(ub)
    reps = []
    for _ in range(5000):
        idx = rng.integers(0, k, k)
        reps.append(S[idx].sum() / C[idx].sum())
    reps = np.array(reps) * 100
    bb[B] = dict(n_blocks=int(k), mean_test_per_block=float(C.mean()),
                 ci95=[float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))], p_le0=float((reps <= 0).mean()))
out['diff_pp'] = float(100 * d.mean())
out['block_bootstrap'] = bb
json.dump(out, open(os.path.join(config.out('base'), 'quick.json'), 'w'), indent=1)
