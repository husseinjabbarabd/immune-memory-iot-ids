import json
import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

K = 11
HS = [0.0025, 0.005, 0.01, 0.02, 0.05]


def z_of(dt):
    return np.arcsinh(dt * 1e6)


def layers(Xa, ya, Xb):
    sigs, sid = np.unique(Xa[:, 1:], axis=0, return_inverse=True)
    smap = {tuple(v): i for i, v in enumerate(sigs)}
    BIG = np.int64(1 << 33)
    OFF = np.int64(1 << 32)
    out = {}
    sidb = np.array([smap.get(tuple(v), -1) for v in Xb[:, 1:]])
    kn = sidb >= 0
    za, zb = z_of(Xa[:, 0]), z_of(Xb[:, 0])
    dv, dinv = np.unique(Xa[:, 0], return_inverse=True)
    ek = sid.astype(np.int64) * BIG + dinv.astype(np.int64)
    ekeys, einv = np.unique(ek, return_inverse=True)
    E = np.zeros((len(ekeys), K), np.float32)
    np.add.at(E, (einv, ya), 1)
    pos = np.clip(np.searchsorted(dv, Xb[kn, 0]), 0, len(dv) - 1)
    hit = dv[pos] == Xb[kn, 0]
    ekb = sidb[kn].astype(np.int64) * BIG + pos
    j = np.clip(np.searchsorted(ekeys, ekb), 0, len(ekeys) - 1)
    ok = hit & (ekeys[j] == ekb)
    L = np.zeros((len(Xb), K), np.float32)
    Lk = np.zeros((kn.sum(), K), np.float32)
    Lk[ok] = E[j[ok]]
    L[kn] = Lk
    out['E'] = L
    for h in HS:
        ba = np.floor(za / h).astype(np.int64)
        bb = np.floor(zb / h).astype(np.int64)
        key = sid.astype(np.int64) * BIG + (ba + OFF)
        keys, inv = np.unique(key, return_inverse=True)
        C = np.zeros((len(keys), K), np.float32)
        np.add.at(C, (inv, ya), 1)
        cum = np.vstack([np.zeros((1, K), np.float32), np.cumsum(C, 0)])
        kb = sidb[kn].astype(np.int64) * BIG + (bb[kn] + OFF)
        lo = np.searchsorted(keys, kb, 'left')
        hi = np.searchsorted(keys, kb, 'right')
        L = np.zeros((len(Xb), K), np.float32)
        L[kn] = cum[hi] - cum[lo]
        out[f'B{h}'] = L
    SIG = np.zeros((len(sigs), K), np.float32)
    np.add.at(SIG, (sid, ya), 1)
    L = np.zeros((len(Xb), K), np.float32)
    L[kn] = SIG[sidb[kn]]
    prior = np.bincount(ya, minlength=K).astype(np.float32)
    L[~kn] = prior
    out['SIG'] = L
    return out


if __name__ == '__main__':
    os.chdir(config.WORK)
    s = np.load('split.npz', allow_pickle=True)
    Xtr, ytr = s['Xtr'], s['ytr']
    folds = list(StratifiedKFold(3).split(Xtr, ytr))
    caches = []
    ys = []
    for tr, va in folds:
        caches.append(layers(Xtr[tr], ytr[tr], Xtr[va]))
        ys.append(ytr[va])
    names = ['E'] + [f'B{h}' for h in HS] + ['SIG']

    def evalc(coef, w):
        a = []
        f = []
        for C, yv in zip(caches, ys):
            S = sum(c * C[n] for c, n in zip(coef, names) if c) * w
            p = S.argmax(1)
            a.append(accuracy_score(yv, p))
            f.append(f1_score(yv, p, average='macro'))
        return np.mean(a), np.mean(f)

    coef = [10, 0, 0, 1, 0, 0, 0.001]
    w = np.ones(K)
    ba, bf = evalc(coef, w)
    obj = ba + 0.5 * bf
    gridc = [0, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32]
    for it in range(3):
        for i in range(len(coef)):
            if names[i] == 'B0.01':
                continue
            best = coef[i]
            for g in (gridc if names[i] != 'SIG' else [0, 0.0005, 0.001, 0.005, 0.02, 0.1, 0.5]):
                c2 = list(coef)
                c2[i] = g
                a, f = evalc(c2, w)
                o = a + 0.5 * f
                if o > obj + 1e-6:
                    obj, best, ba, bf = o, g, a, f
            coef[i] = best
    gridw = [0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4]
    for it in range(3):
        for c in range(K):
            best = w[c]
            for g in gridw:
                w2 = w.copy()
                w2[c] = g
                a, f = evalc(coef, w2)
                o = a + 0.5 * f
                if o > obj + 1e-6:
                    obj, best, ba, bf = o, g, a, f
            w[c] = best
    config.out('base')
    json.dump({'HS': HS, 'names': names, 'coef': list(map(float, coef)), 'w': w.tolist(), 'cv_acc': ba, 'cv_f1m': bf},
              open(config.TUNED, 'w'))
