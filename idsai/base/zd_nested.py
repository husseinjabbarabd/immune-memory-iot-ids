import json
import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from ais import layers, HS, z_of

K = 11
names = ['E'] + [f'B{h}' for h in HS] + ['SIG']
DS = [0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, np.inf]


def dmin_same_sig(Xa, Xb):
    sigs, sid = np.unique(Xa[:, 1:], axis=0, return_inverse=True)
    smap = {tuple(v): i for i, v in enumerate(sigs)}
    za = z_of(Xa[:, 0])
    o = np.lexsort((za, sid))
    sid_s, z_s = sid[o], za[o]
    start = np.searchsorted(sid_s, np.arange(len(sigs)))
    end = np.searchsorted(sid_s, np.arange(len(sigs)) + 1)
    sidb = np.array([smap.get(tuple(v), -1) for v in Xb[:, 1:]])
    zb = z_of(Xb[:, 0])
    out = np.full(len(Xb), np.inf)
    kn = sidb >= 0
    a, b_ = start[sidb[kn]], end[sidb[kn]]
    comp = sid_s.astype(np.float64) * 100.0 + z_s
    ins = np.clip(np.searchsorted(comp, sidb[kn].astype(np.float64) * 100.0 + zb[kn]), a, b_)
    left = np.where(ins > a, np.abs(zb[kn] - z_s[np.maximum(ins - 1, 0)]), np.inf)
    right = np.where(ins < b_, np.abs(z_s[np.minimum(ins, len(z_s) - 1)] - zb[kn]), np.inf)
    out[kn] = np.minimum(left, right)
    return out


def remap_y(y, keep_cls):
    m = np.full(K, -1)
    m[keep_cls] = np.arange(len(keep_cls))
    return m[y]


def tune(X, y, Kc):
    folds = list(StratifiedKFold(3).split(X, y))
    caches = []
    ys = []
    for tr, va in folds:
        L = layers(X[tr], y[tr], X[va])
        caches.append({n: L[n][:, :Kc] for n in names})
        ys.append(y[va])

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
    w = np.ones(Kc, np.float32)
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
        for c in range(Kc):
            best = w[c]
            for g in gridw:
                w2 = w.copy()
                w2[c] = g
                a, f = evalc(coef, w2)
                o = a + 0.5 * f
                if o > obj + 1e-6:
                    obj, best, ba, bf = o, g, a, f
            w[c] = best
    return coef, w, ba, bf


def choose_delta(X, y, Kc, coef, w, nrm):
    folds = list(StratifiedKFold(3).split(X, y))
    cache = []
    for tr, va in folds:
        Xa, ya, Xb, yb = X[tr], y[tr], X[va], y[va]
        for ci in range(Kc):
            if ci == nrm:
                continue
            keep = ya != ci
            kc = [k for k in range(Kc) if k != ci]
            m = np.full(Kc, -1)
            m[kc] = np.arange(len(kc))
            ya2 = m[ya[keep]]
            L = layers(Xa[keep], ya2, Xb)
            S = sum(cf * L[n][:, :len(kc)] for cf, n in zip(coef, names) if cf) * w[kc]
            pred = np.array(kc)[S.argmax(1)]
            cache.append(dict(ci=ci, zd=yb == ci, nm=yb == nrm, pred=pred, dmin=dmin_same_sig(Xa[keep], Xb)))
    res = []
    for d in DS:
        ADR = {}
        FPR = []
        for e in cache:
            eu = e['dmin'] > d
            al = eu | (e['pred'] != nrm)
            ADR.setdefault(e['ci'], []).append(al[e['zd']].mean())
            FPR.append(al[e['nm']].mean())
        res.append((d, 100 * np.mean([np.mean(v) for v in ADR.values()]), 100 * np.mean(FPR)))
    feas = [r for r in res if r[2] <= 2.5]
    return max(feas, key=lambda r: r[1]), res


if __name__ == '__main__':
    os.chdir(config.WORK)
    s = np.load('split.npz', allow_pickle=True)
    Xtr, ytr, Xte, yte = s['Xtr'], s['ytr'], s['Xte'], s['yte']
    classes = [str(c) for c in s['classes']]
    NORMAL = classes.index('Normal')
    out = {}
    for h in range(K):
        if h == NORMAL:
            continue
        keep_cls = [k for k in range(K) if k != h]
        Kc = 10
        m = ytr != h
        X = Xtr[m]
        y = remap_y(ytr[m], keep_cls)
        nrm = keep_cls.index(NORMAL)
        coef, w, cva, cvf = tune(X, y, Kc)
        (dsel, cvADR, cvFPR), sweep = choose_delta(X, y, Kc, coef, w, nrm)
        L = layers(X, y, Xte)
        S = sum(cf * L[n][:, :Kc] for cf, n in zip(coef, names) if cf) * w
        pred = np.array(keep_cls)[S.argmax(1)]
        dm = dmin_same_sig(X, Xte)
        r = dict(coef=[float(c) for c in coef], w=[float(v) for v in w], cv_acc=100 * cva, cv_f1m=100 * cvf,
                 delta=float(dsel), delta_cv_ADR=cvADR, delta_cv_FPR=cvFPR,
                 delta_sweep=[[float(a), b, c] for a, b, c in sweep])
        ht = yte == h
        nt = yte == NORMAL
        att = pred != NORMAL
        for rule, eu in [('strict', np.isinf(dm)), ('temporal', dm > dsel)]:
            r[rule] = dict(EUR=100 * float(eu[ht].mean()), Known=100 * float(((~eu) & att)[ht].mean()),
                           ADR=100 * float((eu | att)[ht].mean()), FPR=100 * float((eu | att)[nt].mean()))
        out[classes[h]] = r
    json.dump(out, open(os.path.join(config.out('base'), 'zd_nested.json'), 'w'), indent=1)
