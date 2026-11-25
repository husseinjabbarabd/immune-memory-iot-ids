import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.chdir(config.WORK)
s = np.load('split.npz', allow_pickle=True)
ytr = s['ytr']
classes = [str(c) for c in s['classes']]
K = 11
N = classes.index('Normal')
rng = np.random.default_rng(21)
nidx = np.where(ytr == N)[0]
rng.shuffle(nidx)
cal = np.zeros(len(ytr), bool)
cal[nidx[320000:]] = True
fi = nidx[:320000]
SYN = [k for k in range(K) if 'SYN' in classes[k]]
rounds = [([h], classes[h]) for h in range(K) if h != N] + [(SYN, 'SYN group')]
log = open(os.path.join(config.out('nested'), 'leak_disjoint.log'), 'w', encoding='utf-8')
ok = True
x = int(cal[fi].sum())
ok &= x == 0
log.write('calibration rows among IF fit rows: %d | OC-SVM fit rows: %d | %s\n' % (x, int(cal[fi[:20000]].sum()),
                                                                                   'PASS' if x == 0 else 'FAIL'))
for hs, hn in rounds:
    fit = ~np.isin(ytr, hs) & ~cal
    n = int((fit & cal).sum())
    ok &= n == 0
    log.write('%s | fit rows %d | calibration rows in fit pool %d | %s\n' % (hn, int(fit.sum()), n, 'PASS' if n == 0 else 'FAIL'))
log.write('ALL PASS\n' if ok else 'FAIL\n')
log.close()
