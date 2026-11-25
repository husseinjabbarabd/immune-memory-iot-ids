import json
import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.chdir(config.WORK)
s = np.load('split.npz', allow_pickle=True)
ytr, itr, ite = s['ytr'], s['itr'], s['ite']
classes = [str(c) for c in s['classes']]
K = 11
N = classes.index('Normal')
R = config.out('nested')
log = open(os.path.join(R, 'leak_audit.log'), 'w', encoding='utf-8')
res = {}


def check(rnd, name, labels, hs):
    n = int(np.isin(labels, hs).sum())
    log.write('%s | %-34s | rows %7d | held-out rows %d | %s\n' % (rnd, name, len(labels), n, 'PASS' if n == 0 else 'FAIL'))
    res.setdefault(rnd, {})[name] = [len(labels), n]
    if n:
        log.close()
        raise SystemExit('leak: %s %s' % (rnd, name))


x = len(np.intersect1d(itr, ite))
log.write('train/test row indices shared: %d %s\n' % (x, 'PASS' if x == 0 else 'FAIL'))
assert x == 0
rng = np.random.default_rng(21)
nidx = np.where(ytr == N)[0]
rng.shuffle(nidx)
cal = np.zeros(len(ytr), bool)
cal[nidx[320000:]] = True
assert (ytr[cal] == N).all()
log.write('calibration partition: %d rows, all Normal: PASS\n' % cal.sum())
fi = nidx[:320000]
SYN = [k for k in range(K) if 'SYN' in classes[k]]
rounds = [([h], classes[h]) for h in range(K) if h != N] + [(SYN, 'SYN group')]
for hs, hn in rounds:
    fit = ~np.isin(ytr, hs) & ~cal
    y = ytr[fit]
    check(hn, 'fit pool (immune memory, XGB refit)', y, hs)
    for i, (tr, va) in enumerate(StratifiedKFold(3).split(np.zeros(len(y)), y)):
        check(hn, 'immune tuning fold %d train' % i, y[tr], hs)
        check(hn, 'immune tuning fold %d validation' % i, y[va], hs)
    a, b = train_test_split(np.arange(len(y)), test_size=0.25, stratify=y, random_state=21)
    check(hn, 'XGB grid fit part', y[a], hs)
    check(hn, 'XGB grid inner validation', y[b], hs)
    check(hn, 'calibration (delta, threshold, lambda)', ytr[cal], hs)
    check(hn, 'IF fit rows', ytr[fi], hs)
    check(hn, 'OC-SVM fit rows', ytr[fi[:20000]], hs)
log.write('ALL PASS\n')
log.close()
json.dump(res, open(os.path.join(R, 'leak_audit.json'), 'w'), indent=1)
