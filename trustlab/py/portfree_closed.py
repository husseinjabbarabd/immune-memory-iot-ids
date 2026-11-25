import csv
import json
import os

import numpy as np
from scipy.io import loadmat

import config

wd = os.path.join(config.RES, 'portfree')
T = loadmat(os.path.join(wd, 'test_rows.mat'), squeeze_me=True)
S = loadmat(os.path.join(wd, 'closed_state.mat'), squeeze_me=True, variable_names=['taus', 'fa', 'popA', 'popM'])
s, p, y = T['sT'].ravel(), T['pAll'].ravel().astype(int), T['yTe'].ravel().astype(int)
taus = np.asarray(S['taus']).ravel()
f1csv = list(csv.DictReader(open(os.path.join(wd, 'closed_e2e_f1_R95.csv'))))
names = [r['names'] for r in f1csv]
ben = names.index('Benign') + 1
C = list(csv.DictReader(open(os.path.join(wd, 'closed_test_results.csv'))))
fa = {}
cnt = {}
isA = y != ben
for o, op in enumerate(('OP-SPEC', 'OP-F2', 'OP-R95')):
    pr = p.copy()
    pr[s < taus[o]] = ben
    n = int(((y == ben) & (pr != ben)).sum())
    fa[op] = {'count': n, 'share': n / len(y)}
    acc = float((pr == y).mean())
    assert abs(acc - float(C[o]['e2e_Acc'])) < 1e-12, (op, acc, C[o]['e2e_Acc'])
    fw = s >= taus[o]
    tp, fp = int((fw & isA).sum()), int((fw & ~isA).sum())
    fn, tn = int((~fw & isA).sum()), int((~fw & ~isA).sum())
    assert tp + fp == int(float(C[o]['nFwd']))
    assert abs(tp / (tp + fn) - float(C[o]['P1_Rec'])) < 1e-12 and abs(tn / (tn + fp) - float(C[o]['P1_Spec'])) < 1e-12
    assert abs((tp + tn) / len(y) - float(C[o]['P1_Acc'])) < 1e-12
    cnt[op] = {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}
assert abs(fa['OP-R95']['share'] - float(S['fa'])) < 1e-15
assert sum(cnt['OP-R95'].values()) == 900334
pr = p.copy()
pr[s < taus[2]] = ben
F0 = {}
for k, nm in enumerate(names, 1):
    tp = ((y == k) & (pr == k)).sum()
    P = tp / max(1, (pr == k).sum())
    R = tp / max(1, (y == k).sum())
    F0[nm] = 2 * P * R / (P + R) if P + R else 0.0
ref = {r['names']: float(r['F1']) for r in f1csv}
assert all(abs(F0[n] - ref[n]) < 1e-12 for n in names)
Q = [r for r in csv.DictReader(open(os.path.join(config.RES, 'study', 'perclass_counts.csv'))) if r['op'] == 'OP-R95']


def i(r, k):
    return int(r[k])


K = sum(i(r, 'n') - i(r, 'fwd') for r in Q if r['class'] != 'Benign')
F1 = {}
for r in Q:
    if r['class'] == 'Benign':
        tp = i(r, 'n') - i(r, 'fwd') + i(r, 'c')
        P = tp / (tp + i(r, 'FPa') + K)
        R = tp / i(r, 'n')
    else:
        P = i(r, 'c') / (i(r, 'c') + i(r, 'FPb') + i(r, 'FPa'))
        R = i(r, 'c') / i(r, 'n')
    F1[r['class']] = 2 * P * R / (P + R)
assert set(F1) == set(names)


def mf(F):
    return float(np.mean([F[n] for n in names]))


assert abs(mf(F0) - float(C[2]['e2e_MF1'])) < 1e-12
json.dump({'fa_portfree': fa, 'counts_portfree': cnt, 'popA': float(S['popA']), 'popM': float(S['popM']),
           'f1_r95_portfree': F0, 'f1_r95_ports': F1, 'mf1_portfree': mf(F0), 'mf1_ports': mf(F1), 'closed': C},
          open(os.path.join(wd, 'closed_extra.json'), 'w'), indent=1)
L = ['port-free false alarms %s: %d (%.3f%%)' % (op, v['count'], 100 * v['share']) for op, v in fa.items()]
L += ['%s TP / FP / FN / TN: %s' % (op, ' / '.join(format(cnt[op][x], ',') for x in ('TP', 'FP', 'FN', 'TN'))) for op in cnt]
L += ['POP-MATCH accuracy %.6f macro F1 %.6f' % (float(S['popA']), float(S['popM']))]
L += ['%-15s F1 ports %.4f  port-free %.4f  diff %+.4f' % (n, F1[n], F0[n], F0[n] - F1[n]) for n in names]
L += ['macro F1 ports %.6f port-free %.6f' % (mf(F1), mf(F0))]
open(os.path.join(wd, 'closed_extra.txt'), 'w').write('\n'.join(L) + '\n')
