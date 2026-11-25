import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

W = config.WORK
OUT = config.out('conflict')
s = np.load(os.path.join(W, 'split.npz'), allow_pickle=True)
Xtr, Xte, yte = s['Xtr'], s['Xte'], s['yte']
P = {'immune': np.load(os.path.join(W, 'final_test_eval.npz'))['pred'],
     'xgb_default': np.load(os.path.join(W, 'xgb_test_pred.npy')),
     'xgb_tuned_rerun': np.load(os.path.join(W, 'xgbT_test_pred.npy')),
     'lookup': np.load(os.path.join(W, 'lookup_test_pred.npy'))}


def rows(X):
    return np.ascontiguousarray(X).view(np.dtype((np.void, X.dtype.itemsize * X.shape[1]))).ravel()


ft, st = rows(Xte), rows(Xte[:, 1:])
seen = np.isin(ft, rows(Xtr))
sig = np.isin(st, rows(Xtr[:, 1:]))
grp = np.where(seen, 0, np.where(sig, 1, 2))
u, inv = np.unique(ft, return_inverse=True)
C = np.zeros((len(u), 11), np.int64)
np.add.at(C, (inv, yte), 1)
nv, mx = C.sum(1), C.max(1)
forced_v = nv - mx
names = ['seen exact vector', 'known signature, new dt', 'unseen signature']
gv = np.zeros(len(u), int)
gv[inv] = grp
out = {'n_test': int(len(yte)), 'ceiling_test_rows': float(mx.sum() / len(yte))}
for name, p in P.items():
    cnt = np.zeros((len(u), 11), np.int64)
    np.add.at(cnt, (inv, p), 1)
    out.setdefault('deterministic_per_vector', {})[name] = bool(((cnt > 0).sum(1) == 1).all())
for g in [0, 1, 2, None]:
    m = np.ones(len(yte), bool) if g is None else grp == g
    vm = np.ones(len(u), bool) if g is None else gv == g
    n = int(m.sum())
    F = int(forced_v[vm].sum())
    d = {'n': n, 'share': n / len(yte), 'conflicted_rows': int(nv[vm & (C > 0).sum(1).astype(bool) & ((C > 0).sum(1) > 1)].sum()),
         'ceiling': (n - F) / n, 'forced_errors': F}
    for name, p in P.items():
        E = int((p[m] != yte[m]).sum())
        d[name] = {'acc': 1 - E / n, 'errors': E, 'forced': F, 'avoidable': E - F,
                   'forced_share_of_errors': F / E if E else None}
    out['all' if g is None else names[g]] = d
json.dump(out, open(os.path.join(OUT, 'conflict_decomp.json'), 'w'), indent=1)
