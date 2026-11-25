import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

s = np.load(os.path.join(config.WORK, 'split.npz'), allow_pickle=True)
Xtr, ytr, Xte, yte = s['Xtr'], s['ytr'], s['Xte'], s['yte']


def rows(X):
    return np.ascontiguousarray(X).view(np.dtype((np.void, X.dtype.itemsize * X.shape[1]))).ravel()


ft, fr = rows(Xte), rows(Xtr)
seen = np.isin(ft, fr)


def multi(keys, labels, q):
    u, inv = np.unique(keys, return_inverse=True)
    C = np.zeros((len(u), 11), np.int64)
    np.add.at(C, (inv, labels), 1)
    m = (C > 0).sum(1) > 1
    pos = np.searchsorted(u, q)
    ok = (pos < len(u))
    ok[ok] = u[pos[ok]] == q[ok]
    r = np.zeros(len(q), bool)
    r[ok] = m[pos[ok]]
    return r


out = {}
for nm, k, l in [('train', fr, ytr), ('full', np.concatenate([fr, ft]), np.concatenate([ytr, yte])), ('test', ft, yte)]:
    r = multi(k, l, ft)
    out[nm] = dict(seen=round(100 * r[seen].mean(), 2), unseen=round(100 * r[~seen].mean(), 2))
json.dump(out, open(os.path.join(config.out('conflict'), 'conflict_share_check.json'), 'w'), indent=1)
