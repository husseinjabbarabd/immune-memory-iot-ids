import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.chdir(config.WORK)
K = 11
s = np.load('split.npz', allow_pickle=True)
ytr, itr, ite = s['ytr'], s['itr'], s['ite']
classes = [str(c) for c in s['classes']]
N = classes.index('Normal')
rng = np.random.default_rng(21)
nidx = np.where(ytr == N)[0]
rng.shuffle(nidx)
cal = np.zeros(len(ytr), bool)
cal[nidx[320000:]] = True


def remap(y, keep):
    m = np.full(K, -1)
    m[keep] = np.arange(len(keep))
    return m[y]


L = ['train/test shared row ids: %d' % len(np.intersect1d(itr, ite)),
     'calibration rows: %d, non-Normal among them: %d' % (cal.sum(), int((ytr[cal] != N).sum()))]
ok = len(np.intersect1d(itr, ite)) == 0 and (ytr[cal] == N).all()
SYN = [k for k in range(K) if 'SYN' in classes[k]]
for hs, hn in [([h], classes[h]) for h in range(K) if h != N] + [(SYN, 'SYN group')]:
    kc = [k for k in range(K) if k not in hs]
    fit = ~np.isin(ytr, hs) & ~cal
    y = remap(ytr[fit], kc)
    yf, yv = train_test_split(y, test_size=0.25, stratify=y, random_state=21)
    c = dict(fit=int(np.isin(ytr[fit], hs).sum()), fit_cal=int((fit & cal).sum()),
             cal=int(np.isin(ytr[cal], hs).sum()),
             grid_fit=int(np.isin(np.array(kc)[yf], hs).sum()), grid_val=int(np.isin(np.array(kc)[yv], hs).sum()))
    good = all(v == 0 for v in c.values())
    ok &= good
    L.append('%-26s fit rows %d | held-out rows: fit %d, calibration %d, grid fit %d, grid validation %d | '
             'fit & calibration %d | %s' % (hn, fit.sum(), c['fit'], c['cal'], c['grid_fit'], c['grid_val'],
                                            c['fit_cal'], 'PASS' if good else 'FAIL'))
L.append('ALL PASS' if ok else 'FAILURES FOUND')
open(os.path.join(config.out('nested'), 'leak_audit_xgbdef.log'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
