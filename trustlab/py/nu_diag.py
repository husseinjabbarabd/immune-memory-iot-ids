import csv
import hashlib
import os

import h5py
import numpy as np
from scipy.io import loadmat

import config
import common_far as u

OUT = config.out('nu_diag')
NAMES, BEN = u.NAMES, u.BEN
PS = (0.5, 0.9, 0.99, 0.995, 0.999)
QN = ['q50', 'q90', 'q99', 'q995', 'q999', 'qmax']
OPS = ('OP-SPEC', 'OP-F2', 'OP-R95')
FLOW_SHA = 'c91ec90b933f612aa3112f3e9082e405ff01d25a71b31b8ea2efa2553923f85f'


def prc(x, p):
    x = np.sort(x)
    n = x.size
    r = (100.0 * p / 100) * n
    if r < 0.5:
        return x[0]
    if r > n - 0.5:
        return x[-1]
    k = int(np.floor(r + 0.5))
    r = r - k
    return (0.5 + r) * x[k] + (0.5 - r) * x[k - 1]


def qs(v):
    return [prc(v, p) for p in PS] + [v.max()]


def diag(ver, rnd, fam, s, q, sv, qv, isX, isB, taus, nu, sav, bad):
    def chk(name, val):
        if not u.same(val, sav[name]):
            bad.append('%s round %s %s %s: %r vs %r' % (ver, rnd, fam, name, val, sav[name]))
    nq = prc(qv, 1 - u.ALPHA)
    if nq != nu:
        bad.append('%s round %s %s: nu %r vs recomputed %r' % (ver, rnd, fam, nu, nq))
    chk('nu', nq)
    QV, QB, QX = qs(qv), qs(q[isB]), qs(q[isX])
    out = []
    for j in range(3):
        t = taus[j]
        farV = np.mean((sv >= t) | (qv >= nu))
        tM = u.match(sv, farV)
        nM = u.match(qv, farV)
        chk('tauM%d' % (j + 1), tM)
        chk('nuM%d' % (j + 1), nM)
        a = s >= t
        o = a | (q >= nu)
        for k, v in (('det_a', a[isX]), ('det_o', o[isX]), ('far_a', a[isB]), ('far_o', o[isB])):
            chk('%s%d' % (k, j + 1), v.mean())
        ex = (q >= nu) & (s < t)
        r = dict(version=ver, round=rnd, family=fam, op=OPS[j], tau=t, nu=nu, nuM=nM, vfar_combined=farV,
                 n_valB=qv.size, n_testB=int(isB.sum()), n_fam=int(isX.sum()))
        for nm, Q, v in (('valB', QV, qv), ('testB', QB, q[isB]), ('fam', QX, q[isX])):
            for qn, val in zip(QN, Q):
                r['%s_%s' % (nm, qn)] = val
            r['%s_ge_nu' % nm] = np.mean(v >= nu)
            r['%s_ge_nuM' % nm] = np.mean(v >= nM)
        for qn, b, c in zip(QN, QB, QV):
            r['shift_%s' % qn] = b - c
        r.update(extra_fam_n=int(ex[isX].sum()), extra_fam_share=ex[isX].mean(), extra_testB_n=int(ex[isB].sum()),
                 extra_testB_share=ex[isB].mean(), det_base=a[isX].mean(), det_combined=o[isX].mean(),
                 far_base=a[isB].mean(), far_combined=o[isB].mean())
        out.append(r)
    return out


def grouped(bad, info):
    wd = os.path.join(config.RES, 'grouped')
    sav = {r['cls']: r for r in csv.DictReader(open(os.path.join(wd, 'zd_original_protocol.csv'), encoding='utf-8'))}
    yTe = loadmat(os.path.join(wd, 'test_rows.mat'), variable_names=['yTe'])['yTe'].ravel().astype(int)
    taus = loadmat(os.path.join(wd, 'closed_state.mat'), variable_names=['taus'])['taus'].ravel().astype(float)
    rows, F = [], None
    for x in range(1, 17):
        if x == BEN:
            continue
        fam = NAMES[x - 1]
        zf = os.path.join(wd, 'zd_%02d.mat' % x)
        if not os.path.exists(zf):
            assert fam not in sav and not np.any(yTe == x), fam
            info.append('grouped %s: no test rows, no round' % fam)
            continue
        Z = loadmat(zf, variable_names=['sx', 'qx', 'nu'])
        d = os.path.join(wd, 'zd%02d' % x)
        if F is None:
            dB = loadmat(os.path.join(d, 'score_test.mat'), variable_names=['dB'])['dB'].ravel().astype(float)
            F = u.feat(dB, Z['qx'].ravel())
        s, q, _ = u.sq(os.path.join(d, 'score_test.mat'), F)
        sv, qv, _ = u.sq(os.path.join(d, 'score_val.mat'), F)
        if not (np.array_equal(s.astype(np.float32), Z['sx'].ravel()) and np.array_equal(q.astype(np.float32), Z['qx'].ravel())):
            bad.append('grouped %s: test scores differ from sx/qx' % fam)
        info.append('grouped %-15s F=%d nValB=%d nTest=%d nX=%d' % (fam, F, sv.size, s.size, int(np.sum(yTe == x))))
        rows += diag('grouped', x, fam, s, q, sv, qv, yTe == x, yTe == BEN, taus, float(Z['nu'].ravel()[0]), sav[fam], bad)
    return rows


def nested(bad, info):
    wd = os.path.join(config.RES, 'nested')
    with h5py.File(config.S09, 'r') as H:
        y = np.array(H['classId']).ravel().astype(int)
    sp = loadmat(os.path.join(config.FS, 'FS_S01', 'FS_S01_split.mat'), variable_names=['split'])['split'].ravel()
    assert hashlib.sha256(np.asarray(sp, np.uint8).tobytes()).hexdigest() == FLOW_SHA
    yTe = y[sp == 3]
    assert np.array_equal(yTe, loadmat(os.path.join(wd, 'test_index.mat'), variable_names=['yTe'])['yTe'].ravel().astype(int))
    yv = y[sp == 2]
    sav = list(csv.DictReader(open(os.path.join(wd, 'rounds.csv'), encoding='utf-8')))
    sav += list(csv.DictReader(open(os.path.join(wd, 'joint.csv'), encoding='utf-8')))
    att = [i for i in range(1, 17) if i != BEN]
    rounds = [[x] for x in att] + [[NAMES.index('DoS') + 1, NAMES.index('Slowloris') + 1]]
    rows = []
    for r, x in enumerate(rounds, 1):
        fam = '+'.join(NAMES[i - 1] for i in x)
        S = sav[r - 1]
        assert int(S['round']) == r and S['family'] == fam, (r, fam)
        L = loadmat(os.path.join(wd, 'round_%02d.mat' % r), variable_names=['sx', 'qx', 'taus', 'nu', 'w'])
        F = int(np.sum(L['w'].ravel() > 0))
        d = os.path.join(wd, 'r%02d' % r)
        s, q, _ = u.sq(os.path.join(d, 'score_test.mat'), F)
        sva, qva, _ = u.sq(os.path.join(d, 'score_val.mat'), F)
        yva = yv[~np.isin(yv, x)]
        assert yva.size == sva.size, (r, yva.size, sva.size)
        vb = yva == BEN
        if not (np.array_equal(s.astype(np.float32), L['sx'].ravel()) and np.array_equal(q.astype(np.float32), L['qx'].ravel())):
            bad.append('random round %d: test scores differ from sx/qx' % r)
        isX = np.isin(yTe, x)
        info.append('random %-15s F=%d nValB=%d nTest=%d nX=%d' % (fam, F, int(vb.sum()), s.size, int(isX.sum())))
        rows += diag('random', r, fam, s, q, sva[vb], qva[vb], isX, yTe == BEN, L['taus'].ravel().astype(float),
                     float(L['nu'].ravel()[0]), S, bad)
    return rows


def fmt(v):
    return '%.15g' % v if isinstance(v, (float, np.floating)) else v


bad, info = [], []
rows = grouped(bad, info) + nested(bad, info)
open(os.path.join(OUT, 'checks.txt'), 'w').write('\n'.join(info + ['mismatches: %d' % len(bad)] + bad) + '\n')
if bad:
    raise SystemExit('checks failed: %d mismatches, see checks.txt' % len(bad))
keys = list(rows[0])
with open(os.path.join(OUT, 'rounds.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, keys)
    w.writeheader()
    for r in rows:
        w.writerow({k: fmt(v) for k, v in r.items()})

L = ['q quantiles: MATLAB prctile formula; shares are fractions of each set.',
     'Saved nu, tau_M and nu_M of every round are recomputed from the validation Benign flows alone (checks.txt).', '']
for ver in ('grouped', 'random'):
    R = [r for r in rows if r['version'] == ver]
    L.append('%s split' % ver)
    L.append('q on validation Benign (vB), test Benign (tB) and held-out family test flows (fam):')
    L.append('%-16s %9s | %9s %9s %9s %9s | %9s %9s %9s %9s | %9s %9s %9s' % (
        'family', 'nu', 'vB q995', 'tB q995', 'tB q999', 'tB qmax', 'vB>=nu', 'tB>=nu', 'fam>=nu', 'fam q50', 'shift50', 'shift995', 'shiftmax'))
    for r in [r for r in R if r['op'] == 'OP-R95']:
        L.append('%-16s %9.6f | %9.6f %9.6f %9.6f %9.6f | %9.5f %9.5f %9.5f %9.6f | %9.6f %9.6f %9.6f' % (
            r['family'], r['nu'], r['valB_q995'], r['testB_q995'], r['testB_q999'], r['testB_qmax'], r['valB_ge_nu'],
            r['testB_ge_nu'], r['fam_ge_nu'], r['fam_q50'], r['shift_q50'], r['shift_q995'], r['shift_qmax']))
    for op in OPS:
        L.append('')
        L.append('%s, %s:' % (ver, op))
        L.append('%-16s %9s %9s | %9s %9s %9s | %8s %9s %8s %9s | %8s %8s' % (
            'family', 'nu', 'nu_M', 'vB>=nuM', 'tB>=nuM', 'fam>=nuM', 'xfam_n', 'xfam', 'xtB_n', 'xtB', 'det_b', 'det_c'))
        for r in [r for r in R if r['op'] == op]:
            L.append('%-16s %9.6f %9.6f | %9.5f %9.5f %9.5f | %8d %9.6f %8d %9.6f | %8.4f %8.4f' % (
                r['family'], r['nu'], r['nuM'], r['valB_ge_nuM'], r['testB_ge_nuM'], r['fam_ge_nuM'], r['extra_fam_n'],
                r['extra_fam_share'], r['extra_testB_n'], r['extra_testB_share'], r['det_base'], r['det_combined']))
        S = [r for r in R if r['op'] == op]
        L.append('rounds %d; rounds with extra family alarms > 0: %d; total extra family alarms %d; total extra test Benign alarms %d; '
                 'rounds with combined = base detection: %d' % (len(S), sum(r['extra_fam_n'] > 0 for r in S), sum(r['extra_fam_n'] for r in S),
                                                                sum(r['extra_testB_n'] for r in S), sum(r['det_base'] == r['det_combined'] for r in S)))
    L.append('')
open(os.path.join(OUT, 'summary.txt'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
