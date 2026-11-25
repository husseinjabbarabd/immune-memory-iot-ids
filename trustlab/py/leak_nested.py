import csv
import os
import sys

import numpy as np
from scipy.io import loadmat, whosmat

import config

wd = os.path.join(config.RES, 'nested')
cnt = {r['className']: r for r in csv.DictReader(open(os.path.join(config.FS, 'FS_S01', 'FS_S01_class_counts.csv'), encoding='utf-8'))}
cid = {n: int(r['classId']) for n, r in cnt.items()}
val_all = sum(int(r['val']) for r in cnt.values())
test_all = sum(int(r['test']) for r in cnt.values())
info = list(csv.DictReader(open(os.path.join(wd, 'rounds.csv'), encoding='utf-8')))
info += list(csv.DictReader(open(os.path.join(wd, 'joint.csv'), encoding='utf-8')))
fam = {int(r['round']): r['family'] for r in info}

L, ok = ['leakage audit, nested rounds', 'validation rows (all classes) %d, test rows %d' % (val_all, test_all), ''], True
for r in range(1, 17):
    x = fam[r].split('+')
    xi = {cid[n] for n in x}
    rd = os.path.join(wd, 'r%02d' % r)
    lab = np.asarray(loadmat(os.path.join(rd, 'airs2.mat'), squeeze_me=True, variable_names=['lab'])['lab']).ravel().astype(int)
    held = int(np.isin(lab, list(xi)).sum())
    miss = sorted(set(cid.values()) - xi - set(lab.tolist()))
    nv = dict((n, s) for n, s, _ in whosmat(os.path.join(rd, 'score_val.mat')))['dA'][0]
    nt = dict((n, s) for n, s, _ in whosmat(os.path.join(rd, 'score_test.mat')))['dA'][0]
    ev = val_all - sum(int(cnt[n]['val']) for n in x)
    o = loadmat(os.path.join(wd, 'round_%02d.mat' % r), squeeze_me=True, struct_as_record=False)['o']
    nX = int(np.asarray(o.nX).ravel()[0])
    eX = sum(int(cnt[n]['test']) for n in x)
    c = [held == 0, not miss, nv == ev, nt == test_all, nX == eX]
    ok &= all(c)
    L.append('%2d %-15s AIRS2 cells %6d, held-out cells %d, classes missing %s | val rows %d (expected %d) | '
             'test rows %d (expected %d) | nX %d (expected %d) | %s'
             % (r, fam[r], lab.size, held, miss or 'none', nv, ev, nt, test_all, nX, eX, 'PASS' if all(c) else 'FAIL'))
L += ['', 'ALL PASS' if ok else 'FAIL (see rows)']
open(os.path.join(wd, 'leak_audit.txt'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
if not ok:
    sys.exit('leakage audit failed')
