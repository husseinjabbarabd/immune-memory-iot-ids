import csv
import json
import os
import sys

import h5py
import numpy as np
import scipy.io as sio

import config

OUT = config.out('study')
FS7 = os.path.join(config.REF, 'FS_S07')
y = np.array(h5py.File(config.S09, 'r')['classId']).ravel().astype(int) - 1
sp = sio.loadmat(os.path.join(config.FS, 'FS_S01', 'FS_S01_split.mat'), variable_names=['split'])['split'].ravel()
y = y[sp == 3]
S = sio.loadmat(os.path.join(FS7, 'FS_S07_TEST_state.mat'), variable_names=['sEval', 'predFsave', 'iFSsave', 'TAUS'])
s = S['sEval'].ravel()
taus = S['TAUS'].ravel()
iF = S['iFSsave'].ravel().astype(int) - 1
pF = S['predFsave'].ravel().astype(int) - 1
B, C = 1, 16
att = [k for k in range(C) if k != B]
names = config.CLASSES
allp = np.full(len(y), B)
allp[iF] = pF
isA = y != B
nB = int((~isA).sum())
nA = int(isA.sum())
wb, wa = 436332 / nB, 597862 / nA
w = np.where(isA, wa, wb)
old = json.load(open(os.path.join(OUT, 'op_metrics.json')))
p1 = json.load(open(os.path.join(OUT, 'reweight.json')))
seal = {r['opPoint']: r for r in csv.DictReader(open(os.path.join(FS7, 'FS_S07_TEST_results.csv')))}
fails = []


def check(name, a, b, tol):
    if abs(a - b) > tol:
        fails.append('%s: %r vs %r' % (name, a, b))


def cm(t, p, v):
    return np.bincount(t * C + p, weights=v, minlength=C * C).reshape(C, C)


def f1(M):
    tp = np.diag(M)
    cs = M.sum(0)
    rs = M.sum(1)
    P = np.divide(tp, cs, out=np.zeros(C), where=cs > 0)
    R = np.divide(tp, rs, out=np.zeros(C), where=rs > 0)
    return P, R, np.divide(2 * P * R, P + R, out=np.zeros(C), where=P + R > 0)


def metrics(v, fw, pr):
    Mf = cm(y[fw], pr[fw], v[fw])
    Ma = cm(y[fw & isA], pr[fw & isA], v[fw & isA])
    Me = cm(y, pr, v)
    _, _, Ff = f1(Mf)
    _, Ra, Fa = f1(Ma)
    _, Re, Fe = f1(Me)
    Pb = np.divide(np.diag(Mf), Mf.sum(0), out=np.zeros(C), where=Mf.sum(0) > 0)
    Fb = np.divide(2 * Pb * Ra, Pb + Ra, out=np.zeros(C), where=Pb + Ra > 0)
    fa = v[~isA & (pr != B)].sum()
    tp = v[fw & isA].sum()
    fp = v[fw & ~isA].sum()
    return dict(fwd16_acc=np.trace(Mf) / Mf.sum(), fwd16_mf1=Ff.mean(), att15_acc=np.trace(Ma) / Ma.sum(),
                att15_mf1_attack_rows=Fa[att].mean(), att15_mf1_with_benign_fp=Fb[att].mean(),
                e2e_acc=np.trace(Me) / Me.sum(), e2e_mf1=Fe.mean(), false_alarm_share=fa / v.sum(),
                p1_rec=tp / v[isA].sum(), p1_spec=1 - fp / v[~isA].sum(), recall_e2e=Re.tolist(), recall_fwd=Ra.tolist()), Me


res = {'weights': {'benign': wb, 'attack': wa, 'total_weight': float(w.sum())}, 'n_test': len(y)}
rows = []
for o, op in enumerate(['OP-SPEC', 'OP-F2', 'OP-R95']):
    fw = s >= np.float32(taus[o])
    pr = np.where(fw, allp, B)
    u, Mu = metrics(np.ones(len(y)), fw, pr)
    r, _ = metrics(w, fw, pr)
    for k in ('fwd16_acc', 'fwd16_mf1', 'att15_acc', 'att15_mf1_attack_rows', 'att15_mf1_with_benign_fp',
              'e2e_acc', 'e2e_mf1', 'false_alarm_share', 'p1_rec', 'p1_spec'):
        check(op + ' unweighted ' + k, u[k], old[op][k], 1e-12)
    check(op + ' rw e2e_acc', r['e2e_acc'], old[op]['e2e_acc_ref_mix'], 1e-12)
    check(op + ' rw fwd16_acc', r['fwd16_acc'], float(seal[op]['fwd_Acc_pc']), 1e-9)
    check(op + ' rw fwd16_mf1', r['fwd16_mf1'], float(seal[op]['fwd_MF1_pc']), 1e-9)
    check(op + ' rw p1_rec', r['p1_rec'], p1[op]['rec'], 1e-12)
    check(op + ' rw p1_spec', r['p1_spec'], p1[op]['spec'], 1e-12)
    for k in ('att15_acc', 'att15_mf1_attack_rows', 'p1_rec', 'p1_spec'):
        check(op + ' unchanged ' + k, r[k], u[k], 1e-12)
    for i in range(C):
        check(op + ' unchanged recall_e2e ' + names[i], r['recall_e2e'][i], u['recall_e2e'][i], 1e-12)
        if i != B:
            check(op + ' unchanged recall_fwd ' + names[i], r['recall_fwd'][i], u['recall_fwd'][i], 1e-12)
    if op == 'OP-R95':
        st = [[int(x) for x in row[1:]] for row in list(csv.reader(open(os.path.join(FS7, 'FS_S07_TEST_confusion_R95.csv'))))[1:]]
        if not np.array_equal(Mu.astype(int), np.array(st)):
            fails.append('OP-R95 end-to-end matrix differs from FS_S07_TEST_confusion_R95.csv')
    res[op] = {'reweighted': {k: r[k] for k in ('att15_mf1_with_benign_fp', 'e2e_mf1', 'false_alarm_share')},
               'reweighted_reproduced': {k: r[k] for k in ('fwd16_acc', 'fwd16_mf1', 'e2e_acc')},
               'unchanged_by_reweighting': {k: r[k] for k in ('att15_acc', 'att15_mf1_attack_rows', 'p1_rec', 'p1_spec')}}
    for i in range(C):
        f = fw & (y == i)
        rows.append([op, names[i], int((y == i).sum()), int(f.sum()), int((f & (pr == i)).sum()),
                     int((fw & ~isA & (pr == i)).sum()) if i != B else 0,
                     int((fw & isA & (y != i) & (pr == i)).sum())])
if fails:
    sys.exit('check failed:\n' + '\n'.join(fails))
with open(os.path.join(OUT, 'perclass_counts.csv'), 'w', newline='') as fh:
    wr = csv.writer(fh)
    wr.writerow(['op', 'class', 'n', 'fwd', 'c', 'FPb', 'FPa'])
    wr.writerows(rows)
json.dump(res, open(os.path.join(OUT, 'reweight_all.json'), 'w'), indent=1, default=float)
