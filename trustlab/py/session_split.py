import json
import os

import numpy as np
import h5py
import scipy.io as sio

import config

S09, NAMES = config.S09, config.NAMES
OUT = os.path.join(config.out('splits'), 'split_sessions')

ben = int(sio.loadmat(NAMES)['benignIdx'].ravel()[0])
with h5py.File(S09, 'r') as H:
    ts = H['ts'][()].ravel()
    y = H['classId'][()].ravel().astype(int)
    fid = H['fileId'][()].ravel().astype(int)
n = len(y)
C = [l.split(',') for l in open(config.COUNTS).read().split('\n')[1:] if l]
names = [c[0].strip('"') for c in C]
assert np.bincount(y)[1:].tolist() == [int(c[1]) for c in C] and names[ben - 1] == 'Benign'


def sessions(t, g):
    o = np.argsort(t, kind='stable')
    s = np.concatenate([[0], np.cumsum(np.diff(t[o]) > g)])
    return o, s


split = np.zeros(n, np.uint8)
rows = []
for k in range(1, len(names) + 1):
    idx = np.flatnonzero(y == k)
    t = ts[idx]
    nk = len(idx)
    cnt = {g: int(sessions(t, g)[1][-1] + 1) for g in (10, 60, 300)}
    o, s = sessions(t, 60)
    ns = s[-1] + 1
    size = np.bincount(s, minlength=ns)
    cum = np.cumsum(size) / nk
    sub = np.where(np.concatenate([[0], cum[:-1]]) < 0.64, 1, np.where(np.concatenate([[0], cum[:-1]]) < 0.80, 2, 3))
    share = [size[sub == j].sum() / nk for j in (1, 2, 3)]
    grp = ns >= 3 and min(share) >= 0.05
    lab = np.zeros(nk, np.uint8)
    emb = 0
    if grp:
        lab[o] = sub[s]
        mode = 'grouped'
    else:
        mode = 'time-block'
        r = np.arange(nk) / nk
        lab[o] = np.where(r < 0.64, 1, np.where(r < 0.80, 2, 3))
        to = t[o]
        for j in (2, 3):
            first = np.flatnonzero(lab[o] == j)
            if len(first):
                b0 = to[first[0]]
                cut = first[to[first] < b0 + 5]
                emb += len(cut)
                lab[o[cut]] = 0
    split[idx] = lab
    rows.append(dict(cls=names[k - 1], rows=nk, s10=cnt[10], s60=cnt[60], s300=cnt[300], mode=mode,
                     train=int((lab == 1).sum()), val=int((lab == 2).sum()), test=int((lab == 3).sum()),
                     embargo=emb, t0=float(t.min()), t1=float(t.max()), files=int(len(np.unique(fid[idx])))))

P1, P2 = np.uint64(0x9E3779B97F4A7C15), np.uint64(0xC2B2AE3D27D4EB4F)
h1 = np.zeros(n, np.uint64)
h2 = np.zeros(n, np.uint64)
with h5py.File(S09, 'r') as H:
    X = H['X']
    for a in range(0, n, 200000):
        b = min(a + 200000, n)
        V = np.ascontiguousarray(X[:, a:b].T).view(np.uint32).astype(np.uint64)
        x1 = np.zeros(b - a, np.uint64)
        x2 = np.zeros(b - a, np.uint64)
        with np.errstate(over='ignore'):
            for j in range(V.shape[1]):
                x1 = x1 * P1 + V[:, j] + np.uint64(1)
                x2 = (x2 ^ V[:, j]) * P2 + np.uint64(j + 7)
        h1[a:b] = x1
        h2[a:b] = x2
key = h1 ^ (h2 << np.uint64(1))
trk = set(key[split == 1].tolist())
for r, k in zip(rows, range(1, len(names) + 1)):
    te = (y == k) & (split == 3)
    r['test_in_train'] = float(np.mean([v in trk for v in key[te].tolist()])) if te.any() else None

json.dump(dict(gap=60, rows=rows, benign=names[ben - 1], n=int(n),
               removed=int((split == 0).sum())), open(OUT + '.json', 'w'), indent=1)
np.save(OUT + '.npy', split)
