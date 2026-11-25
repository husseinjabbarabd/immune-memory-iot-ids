import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from ais import K, HS, z_of

BIG = np.int64(1 << 33)
OFF = np.int64(1 << 32)


def build(Xa, ya):
    sigs, sid = np.unique(Xa[:, 1:], axis=0, return_inverse=True)
    za = z_of(Xa[:, 0])
    dv, dinv = np.unique(Xa[:, 0], return_inverse=True)
    ekeys, einv = np.unique(sid.astype(np.int64) * BIG + dinv.astype(np.int64), return_inverse=True)
    E = np.zeros((len(ekeys), K), np.float32)
    np.add.at(E, (einv, ya), 1)
    M = {'sigs': sigs, 'dv': dv, 'ekeys': ekeys, 'E': E}
    for i, h in enumerate(HS):
        keys, inv = np.unique(sid.astype(np.int64) * BIG + (np.floor(za / h).astype(np.int64) + OFF), return_inverse=True)
        C = np.zeros((len(keys), K), np.float32)
        np.add.at(C, (inv, ya), 1)
        M['bkeys%d' % i], M['B%d' % i] = keys, C
    SIG = np.zeros((len(sigs), K), np.float32)
    np.add.at(SIG, (sid, ya), 1)
    M['SIG'] = SIG
    M['prior'] = np.bincount(ya, minlength=K).astype(np.float32)
    return M


def find(keys, q):
    j = np.clip(np.searchsorted(keys, q), 0, len(keys) - 1)
    return j, keys[j] == q


def layers(M, Xb):
    smap = {tuple(v): i for i, v in enumerate(M['sigs'])}
    sidb = np.array([smap.get(tuple(v), -1) for v in Xb[:, 1:]])
    kn = sidb >= 0
    zb = z_of(Xb[:, 0])
    out = {}
    dv = M['dv']
    pos = np.clip(np.searchsorted(dv, Xb[kn, 0]), 0, len(dv) - 1)
    j, ok = find(M['ekeys'], sidb[kn].astype(np.int64) * BIG + pos)
    ok &= dv[pos] == Xb[kn, 0]
    L = np.zeros((len(Xb), K), np.float32)
    Lk = np.zeros((kn.sum(), K), np.float32)
    Lk[ok] = M['E'][j[ok]]
    L[kn] = Lk
    out['E'] = L
    for i, h in enumerate(HS):
        j, ok = find(M['bkeys%d' % i], sidb[kn].astype(np.int64) * BIG + (np.floor(zb[kn] / h).astype(np.int64) + OFF))
        L = np.zeros((len(Xb), K), np.float32)
        Lk = np.zeros((kn.sum(), K), np.float32)
        Lk[ok] = M['B%d' % i][j[ok]]
        L[kn] = Lk
        out[f'B{h}'] = L
    L = np.zeros((len(Xb), K), np.float32)
    L[kn] = M['SIG'][sidb[kn]]
    L[~kn] = M['prior']
    out['SIG'] = L
    return out


if __name__ == '__main__':
    s = np.load(os.path.join(config.WORK, 'split.npz'), allow_pickle=True)
    Xtr, ytr, Xte = s['Xtr'], s['ytr'], s['Xte']
    c = json.load(open(config.TUNED))
    M = build(Xtr, ytr)
    w = np.array(c['w'], np.float32)
    L = layers(M, Xte)
    p = (sum(cf * L[n] for cf, n in zip(c['coef'], c['names']) if cf) * w).argmax(1)
    ref = np.load(os.path.join(config.WORK, 'final_test_eval.npz'))['pred']
    assert np.array_equal(p, ref), 'stored model does not give the test predictions'
    np.savez_compressed(os.path.join(config.out('base'), 'model.npz'), coef=np.array(c['coef']), w=w,
                        names=np.array(c['names']), HS=np.array(HS), classes=s['classes'], **M)
