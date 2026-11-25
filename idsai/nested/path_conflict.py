import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from ais import layers
from zd_nested import names, remap_y

os.chdir(config.WORK)
K = 11
s = np.load('split.npz', allow_pickle=True)
Xtr, ytr, Xte, yte = s['Xtr'], s['ytr'], s['Xte'], s['yte']
classes = [str(c) for c in s['classes']]
N = classes.index('Normal')
assert not np.isnan(Xtr).any() and not np.isnan(Xte).any()
rng = np.random.default_rng(21)
nidx = np.where(ytr == N)[0]
rng.shuffle(nidx)
cal = np.zeros(len(ytr), bool)
cal[nidx[320000:]] = True
assert (ytr[cal] == N).all()
Xc = Xtr[cal]

R = config.out('nested')
base = json.load(open(os.path.join(R, 'nested_fpr.json')))
wn = json.load(open(os.path.join(R, 'nested_wn.json')))
out = {'meta': dict(rule='(a) max attack count > Normal count among identical memory vectors; '
                         '(b1) vector in memory, Normal >= every attack; (b2) vector absent')}


def key(A):
    A = np.ascontiguousarray(A + 0.0)
    return A.view(np.dtype((np.void, A.dtype.itemsize * A.shape[1]))).ravel()


SYN = [k for k in range(K) if 'SYN' in classes[k]]
rounds = [([h], classes[h]) for h in range(K) if h != N] + [(SYN, 'SYN group')]
for hs, hn in rounds:
    bi = base[hn]['immune']
    kc = [k for k in range(K) if k not in hs]
    Kc = len(kc)
    kc_ = np.array(kc)
    nr = kc.index(N)
    fit = ~np.isin(ytr, hs) & ~cal
    X = Xtr[fit]
    y = remap_y(ytr[fit], kc)
    assert not np.isin(ytr[fit], hs).any() and not (fit & cal).any()
    coef = bi['coef']
    w = np.array(bi['w'], np.float32)

    def pred(Xb):
        L = layers(X, y, Xb)
        return kc_[(sum(cf * L[n][:, :Kc] for cf, n in zip(coef, names) if cf) * w).argmax(1)]

    pc, pt = pred(Xc), pred(Xte)
    ht = np.isin(yte, hs)
    nt = yte == N
    g = dict(path_cal_fpr=100 * float((pc != N).mean()), path_test_fpr=100 * float((pt != N)[nt].mean()),
             seen_acc=100 * float((pt == yte)[~ht].mean()))
    c = wn[hn]['immune']['curve_m1'][-1]
    ref = dict(path_cal_fpr=bi['path_cal_fpr'], path_cal_curve=c[0], path_test_fpr=c[1], seen_acc=wn[hn]['immune']['base_seen_acc'])
    if not all(abs(g[q.replace('path_cal_curve', 'path_cal_fpr')] - v) < 1e-9 for q, v in ref.items()):
        out[hn] = dict(gate=g, stored=ref, reproduced=False)
        continue
    um, inv = np.unique(key(X), return_inverse=True)
    cnt = np.zeros((len(um), Kc), np.int64)
    np.add.at(cnt, (inv.ravel(), y), 1)
    ca = cnt.copy()
    ca[:, nr] = -1
    forced = ca.max(1) > cnt[:, nr]
    idx = {k.tobytes(): i for i, k in enumerate(um)}

    def split(Q, p):
        j = np.array([idx.get(k.tobytes(), -1) for k in key(Q)])
        fa = p != N
        inm = j >= 0
        a = fa & inm & forced[np.maximum(j, 0)]
        b1 = fa & inm & ~a
        b2 = fa & ~inm
        nfa = int(fa.sum())
        return dict(normal_rows=len(Q), false_alarms=nfa, fa_pct=100 * nfa / len(Q),
                    a_forced=int(a.sum()), b1_normal_majority=int(b1.sum()), b2_absent=int(b2.sum()),
                    a_share=100 * float(a.sum()) / max(nfa, 1), b1_share=100 * float(b1.sum()) / max(nfa, 1),
                    b2_share=100 * float(b2.sum()) / max(nfa, 1),
                    normal_rows_attack_majority=int((inm & forced[np.maximum(j, 0)]).sum()))

    out[hn] = dict(gate=g, reproduced=True, memory_rows=int(len(X)), memory_vectors=int(len(um)),
                   calibration=split(Xc, pc), test=split(Xte[nt], pt[nt]))
json.dump(out, open(os.path.join(R, 'path_conflict.json'), 'w'), indent=1)
