import json
import os

import h5py
import numpy as np
import scipy.io as sio

import config

OUT = config.out('study')
y = np.array(h5py.File(config.S09, 'r')['classId']).ravel().astype(int) - 1
sp = sio.loadmat(os.path.join(config.FS, 'FS_S01', 'FS_S01_split.mat'), variable_names=['split'])['split'].ravel()
y = y[sp == 3]
S = sio.loadmat(os.path.join(config.REF, 'FS_S07', 'FS_S07_TEST_state.mat'),
                variable_names=['sEval', 'predFsave', 'iFSsave', 'TAUS'])
s = S['sEval'].ravel()
taus = S['TAUS'].ravel()
iF = S['iFSsave'].ravel().astype(int) - 1
pF = S['predFsave'].ravel().astype(int) - 1
B = 1
C = 16
att = [k for k in range(C) if k != B]
NB, NA = 436332, 597862
allp = np.full(len(y), B)
allp[iF] = pF
isA = y != B
nB = int((~isA).sum())
nA = int(isA.sum())


def cm(t, p):
    return np.bincount(t * C + p, minlength=C * C).reshape(C, C)


def f1s(M):
    P = np.zeros(C)
    R = np.zeros(C)
    F = np.zeros(C)
    for k in range(C):
        tp = M[k, k]
        cs = M[:, k].sum()
        rs = M[k].sum()
        P[k] = tp / cs if cs else 0
        R[k] = tp / rs if rs else 0
        F[k] = 2 * P[k] * R[k] / (P[k] + R[k]) if P[k] + R[k] else 0
    return P, R, F


res = {'n_test': len(y), 'n_benign': nB, 'n_attack': nA, 'per_class': np.bincount(y, minlength=C).tolist()}
for o, name in enumerate(['OP-SPEC', 'OP-F2', 'OP-R95']):
    fw = s >= np.float32(taus[o])
    pr = np.where(fw, allp, B)
    Mf = cm(y[fw], pr[fw])
    Ma = cm(y[fw & isA], pr[fw & isA])
    Me = cm(y, pr)
    _, _, Ff = f1s(Mf)
    _, _, Fa = f1s(Ma)
    _, _, Fe = f1s(Me)
    Pb = np.array([Mf[k, k] / Mf[:, k].sum() if Mf[:, k].sum() else 0 for k in range(C)])
    Rb = np.array([Ma[k, k] / Ma[k].sum() if Ma[k].sum() else 0 for k in range(C)])
    Fb = np.where(Pb + Rb > 0, 2 * Pb * Rb / np.where(Pb + Rb > 0, Pb + Rb, 1), 0)
    tp = int((fw & isA).sum())
    fp = int((fw & ~isA).sum())
    fn = nA - tp
    tn = nB - fp
    w = np.where(isA, NA / nA, NB / nB)
    ok = pr == y
    res[name] = dict(
        tau=float(taus[o]), forwarded=int(fw.sum()), TP=tp, FP=fp, FN=fn, TN=tn,
        p1_acc=(tp + tn) / len(y), p1_rec=tp / nA, p1_spec=tn / nB, p1_prec=tp / (tp + fp),
        p1_f2=5 * tp / (5 * tp + 4 * fn + fp),
        fwd16_n=int(fw.sum()), fwd16_acc=np.trace(Mf) / Mf.sum(), fwd16_mf1=Ff.mean(),
        att15_n=tp, att15_acc=np.trace(Ma) / Ma.sum(), att15_mf1_attack_rows=Fa[att].mean(),
        att15_mf1_with_benign_fp=Fb[att].mean(),
        e2e_acc=ok.mean(), e2e_mf1=Fe.mean(), e2e_acc_ref_mix=float((w * ok).sum() / w.sum()),
        false_alarms=int((~isA & (pr != B)).sum()), false_alarm_share=float((~isA & (pr != B)).sum() / len(y)))
json.dump(res, open(os.path.join(OUT, 'op_metrics.json'), 'w'), indent=1, default=float)
