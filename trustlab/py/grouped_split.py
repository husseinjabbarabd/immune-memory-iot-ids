import json
import os

import numpy as np
import h5py
import scipy.io as sio

import config

S09, NAMES = config.S09, config.NAMES
OUT = os.path.join(config.out('splits'), 'split_grouped')
GAP, TR, VAL = 60.0, 0.64, 0.16
LO, HI, MIN = 0.40, 0.80, 0.05
EMB, EMBF, EMBCAP = 5.0, 0.01, 0.02

ben = int(sio.loadmat(NAMES)['benignIdx'].ravel()[0])
with h5py.File(S09, 'r') as H:
    ts = H['ts'][()].ravel()
    y = H['classId'][()].ravel().astype(int)
n = len(y)
C = [l.split(',') for l in open(config.COUNTS).read().split('\n')[1:] if l]
names = [c[0].strip('"') for c in C]
assert np.bincount(y)[1:].tolist() == [int(c[1]) for c in C] and names[ben - 1] == 'Benign'


def tcut(t, i):
    while 0 < i < len(t) and t[i] == t[i - 1]:
        i += 1
    return i


def cut(t, sb, a, b, want, lo, hi):
    nk = len(t)
    c = [j for j in sb if a < j < b and lo <= (j - a) / nk <= hi]
    if c:
        return min(c, key=lambda j: abs((j - a) / nk - want)), 'session'
    return tcut(t, a + int(round(want * nk))), 'time'


split = np.zeros(n, np.uint8)
rows = []
for k in range(1, len(names) + 1):
    idx = np.flatnonzero(y == k)
    o = np.argsort(ts[idx], kind='stable')
    t = ts[idx][o]
    nk = len(t)
    sb = (np.flatnonzero(np.diff(t) > GAP) + 1).tolist()
    c1, m1 = cut(t, sb, 0, nk, TR, LO, HI)
    c2, m2 = cut(t, sb, c1, nk, (nk - c1) / nk * VAL / (1 - TR), MIN, (nk - c1) / nk - MIN)
    lab = np.ones(nk, np.uint8)
    lab[c1:c2] = 2
    lab[c2:] = 3
    g = min(EMB, EMBF * (t[-1] - t[0]))
    emb = []
    for cc, m in ((c1, m1), (c2, m2)):
        if m == 'time':
            e = cc + int(np.searchsorted(t[cc:], t[cc] + g, 'right'))
            e = min(e, cc + int(EMBCAP * nk))
            lab[cc:e] = 0
            emb.append(e - cc)
        else:
            emb.append(0)
    split[idx[o]] = lab
    cnt = np.bincount(lab, minlength=4)
    assert min(cnt[1:]) >= (MIN - EMBCAP) * nk, names[k - 1]
    rows.append(dict(cls=names[k - 1], rows=nk, sessions=len(sb) + 1, span=float(t[-1] - t[0]), mode=m1 + '/' + m2,
                     train=int(cnt[1]), val=int(cnt[2]), test=int(cnt[3]), emb_tv=emb[0], emb_vt=emb[1],
                     gap_s=float(g) if 'time' in (m1, m2) else None,
                     gap_tv=float(t[c1] - t[c1 - 1]) if m1 == 'session' else None,
                     gap_vt=float(t[c2] - t[c2 - 1]) if m2 == 'session' else None))

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
    r['test_in_train'] = float(np.mean([v in trk for v in key[te].tolist()]))

json.dump(dict(rows=rows, n=int(n), removed=int((split == 0).sum()),
               counts=np.bincount(split, minlength=4).tolist()), open(OUT + '.json', 'w'), indent=1)
np.save(OUT + '.npy', split)
sio.savemat(OUT + '.mat', {'gsplit': split.reshape(-1, 1)})
