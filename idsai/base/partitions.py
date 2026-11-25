import os
import sys

import numpy as np
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.chdir(config.WORK)
R = config.out('partitions')


def write(name, idx):
    idx = np.sort(np.asarray(idx, np.int64))
    with open(os.path.join(R, name + '.txt'), 'w', newline='\n') as f:
        f.write(''.join('%d\n' % i for i in idx))


s = np.load('split.npz', allow_pickle=True)
itr, ite, ytr = s['itr'], s['ite'], s['ytr']
classes = [str(c) for c in s['classes']]
assert len(np.intersect1d(itr, ite)) == 0
write('train', itr)
write('test', ite)
N = classes.index('Normal')
rng = np.random.default_rng(21)
nidx = np.where(ytr == N)[0]
rng.shuffle(nidx)
cal = itr[nidx[320000:]]
assert len(cal) == 80000
write('calib', cal)
d = np.load('idsai19.npz')
y = LabelEncoder().fit_transform(d['y'])
assert np.array_equal(y[ite], s['yte']) and np.array_equal(y[itr], ytr)
tr = []
te = []
for k in range(len(classes)):
    i = np.flatnonzero(y == k)
    n = len(i) - int(round(0.2 * len(i)))
    tr.append(i[:n])
    te.append(i[n:])
tr, te = np.sort(np.concatenate(tr)), np.sort(np.concatenate(te))
assert np.array_equal(te, np.load('blocked_preds.npz')['te'])
write('blocked_train', tr)
write('blocked_test', te)
