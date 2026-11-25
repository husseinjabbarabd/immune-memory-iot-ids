import json
import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from ais import layers
from zd_nested import names, dmin_same_sig, remap_y, tune

os.chdir(config.WORK)
K = 11
s = np.load('split.npz', allow_pickle=True)
Xtr, ytr, Xte, yte = s['Xtr'], s['ytr'], s['Xte'], s['yte']
classes = [str(c) for c in s['classes']]
N = classes.index('Normal')
T = [0.005, 0.01, 0.02, 0.05]
DG = np.r_[np.geomspace(1e-4, 10, 200), np.inf]
M = [round(1 + 0.05 * i, 2) for i in range(21)]

rng = np.random.default_rng(21)
nidx = np.where(ytr == N)[0]
rng.shuffle(nidx)
cal = np.zeros(len(ytr), bool)
cal[nidx[320000:]] = True
Xc = Xtr[cal]
FG = np.round(np.arange(1, 201) * 0.0005, 4)

R = config.out('nested')
base = json.load(open(os.path.join(R, 'nested_fpr.json')))
out = {'meta': dict(m_grid=M, targets=T)}
audit = open(os.path.join(R, 'leak_audit_wn.log'), 'w', encoding='utf-8')


def leak(rnd, name, labels, hs):
    n = int(np.isin(labels, hs).sum())
    audit.write('%s | %-30s | rows %7d | held-out rows %d | %s\n' % (rnd, name, len(labels), n, 'PASS' if n == 0 else 'FAIL'))
    if n:
        audit.close()
        raise SystemExit('leak: %s %s' % (rnd, name))


assert len(np.intersect1d(s['itr'], s['ite'])) == 0 and (ytr[cal] == N).all()
audit.write('train/test disjoint, calibration all Normal (%d rows): PASS\n' % cal.sum())


def curve(calfpr, alarm, flag, ht, nt):
    return [100 * float(calfpr), 100 * float(alarm[nt].mean()), 100 * float(alarm[ht].mean()), 100 * float(flag[ht].mean())]


def same(a, b, what):
    if not np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=0, atol=1e-9):
        raise SystemExit('reproduction failed: %s %s != %s' % (what, a, b))


NJB = {'SYN group': 16}


def xgb(d, lr, ne, nj=2):
    return XGBClassifier(eval_metric='mlogloss', n_jobs=nj, colsample_bytree=0.75, gamma=1.5, max_depth=d,
                         min_child_weight=1, subsample=1, learning_rate=lr, n_estimators=ne, tree_method='exact',
                         random_state=0)


def metrics(alarm, flag, pred, hs, ht, nt):
    seen = ~ht
    return dict(EUR=100 * float(flag[ht].mean()), ADR=100 * float(alarm[ht].mean()), FPR=100 * float(alarm[nt].mean()),
                per_class={classes[k]: [100 * float(flag[yte == k].mean()), 100 * float(alarm[yte == k].mean())] for k in hs},
                seen_acc=100 * float((pred == yte)[seen].mean()), seen_attack_alarm=100 * float(alarm[seen & ~nt].mean()))


SYN = [k for k in range(K) if 'SYN' in classes[k]]
rounds = [([h], classes[h]) for h in range(K) if h != N] + [(SYN, 'SYN group')]
for hs, hn in rounds:
    b = base[hn]
    r = out.setdefault(hn, {})
    kc = [k for k in range(K) if k not in hs]
    Kc = len(kc)
    kc_ = np.array(kc)
    fit = ~np.isin(ytr, hs) & ~cal
    X = Xtr[fit]
    y = remap_y(ytr[fit], kc)
    nr = kc.index(N)
    ht = np.isin(yte, hs)
    nt = yte == N
    yo = ytr[fit]
    leak(hn, 'fit pool (memory, XGB refit)', yo, hs)
    leak(hn, 'calibration (delta, th, lambda)', ytr[cal], hs)
    for i, (tr, va) in enumerate(StratifiedKFold(3).split(X, y)):
        leak(hn, 'immune tuning fold %d train' % i, yo[tr], hs)
        leak(hn, 'immune tuning fold %d val' % i, yo[va], hs)
    bi = b['immune']
    coef, w, _, _ = tune(X, y, Kc)
    same(coef, bi['coef'], hn + ' coef')
    same(w, bi['w'], hn + ' w')

    def raw(Xb):
        L = layers(X, y, Xb)
        return sum(cf * L[n][:, :Kc] for cf, n in zip(coef, names) if cf)

    Sc, St = raw(Xc), raw(Xte)
    dc, dt = dmin_same_sig(X, Xc), dmin_same_sig(X, Xte)

    def pred(S, m):
        ww = np.array(w).copy()
        ww[nr] = ww[nr] * m
        return kc_[(S * ww).argmax(1)]

    pfpr = {m: float((pred(Sc, m) != N).mean()) for m in M}
    same(100 * pfpr[1.0], bi['path_cal_fpr'], hn + ' immune path')
    p1 = pred(St, 1.0)
    im = dict(path_cal_fpr_by_m={str(m): 100 * v for m, v in pfpr.items()},
              base_seen_acc=100 * float((p1 == yte)[~ht].mean()))
    for f in T:
        ok_m = [m for m in M if pfpr[m] <= f]
        if not ok_m:
            im[str(f)] = dict(attainable=False, path_cal_fpr_at_max=100 * pfpr[M[-1]])
            continue
        m = ok_m[0]
        attc = pred(Sc, m) != N
        cf_ = np.array([((dc > d) | attc).mean() for d in DG])
        d = DG[np.where(cf_ <= f)[0].min()]
        pt = p1 if m == 1.0 else pred(St, m)
        eu = dt > d
        im[str(f)] = dict(attainable=True, m=m, path_cal_fpr=100 * pfpr[m], delta=float(d),
                          cal_fpr=100 * float(cf_[np.where(cf_ <= f)[0].min()]), **metrics(eu | (pt != N), eu, pt, hs, ht, nt))
        if m == 1.0:
            for k in ('delta', 'cal_fpr', 'EUR', 'ADR', 'FPR'):
                same(im[str(f)][k], bi[str(f)][k], hn + ' immune ' + k)
    a1c, a1t = pred(Sc, 1.0) != N, p1 != N
    im['curve_m1'] = [curve(((dc > d) | a1c).mean(), (dt > d) | a1t, dt > d, ht, nt) for d in DG]
    im['base_seen_attack_alarm'] = {str(f): 100 * float(((dt > bi[str(f)]['delta']) | (p1 != N))[~ht & ~nt].mean())
                                    for f in T if bi[str(f)]['attainable']}
    im['reproduced'] = True
    r['immune'] = im
    bx = b['xgb']
    mdl = xgb(*bx['best'], nj=NJB.get(hn, 2)).fit(X, y)
    Pc, Pt = mdl.predict_proba(Xc), mdl.predict_proba(Xte)

    def pred(P, m):
        ww = np.ones(Kc)
        ww[nr] = m
        return (P * ww).argmax(1)

    def conf(P, p):
        return np.where(p == nr, P[:, nr], P.max(1))

    pfpr = {m: float((pred(Pc, m) != nr).mean()) for m in M}
    same(round(pfpr[1.0] * len(Pc)), bx['path_cal_rows'], hn + ' xgb path rows')
    p1 = pred(Pt, 1.0)
    xr = dict(path_cal_fpr_by_m={str(m): 100 * v for m, v in pfpr.items()},
              base_seen_acc=100 * float((kc_[p1] == yte)[~ht].mean()))
    base_alarm = {}
    for f in T:
        if bx[str(f)]['attainable']:
            base_alarm[str(f)] = 100 * float(((conf(Pt, p1) < bx[str(f)]['th']) | (p1 != nr))[~ht & ~nt].mean())
        ok_m = [m for m in M if pfpr[m] <= f]
        if not ok_m:
            xr[str(f)] = dict(attainable=False, path_cal_fpr_at_max=100 * pfpr[M[-1]])
            continue
        m = ok_m[0]
        pc = pred(Pc, m)
        cc = conf(Pc, pc)
        need = f - pfpr[m]
        if need == 0:
            th = 0.0
        else:
            ms = np.sort(cc[pc == nr])
            k = int(round(need * len(Pc)))
            th = float(ms[k - 1]) + 1e-12 if k > 0 else 0.0
        pt = p1 if m == 1.0 else pred(Pt, m)
        fl = conf(Pt, pt) < th
        xr[str(f)] = dict(attainable=True, m=m, path_cal_fpr=100 * pfpr[m], th=th,
                          cal_fpr=100 * float(((cc < th) | (pc != nr)).mean()),
                          **metrics(fl | (pt != nr), fl, kc_[pt], hs, ht, nt))
        if m == 1.0:
            for k in ('th', 'cal_fpr', 'EUR', 'ADR', 'FPR'):
                same(xr[str(f)][k], bx[str(f)][k], hn + ' xgb ' + k)
    pc1 = pred(Pc, 1.0)
    c1 = conf(Pc, pc1)
    ms1 = np.sort(c1[pc1 == nr])
    ct1 = conf(Pt, p1)
    cv = []
    for f in FG:
        need = f - pfpr[1.0]
        if need < 0:
            continue
        k = int(round(need * len(Pc)))
        th = float(ms1[k - 1]) + 1e-12 if k > 0 else 0.0
        cv.append(curve(((c1 < th) | (pc1 != nr)).mean(), (ct1 < th) | (p1 != nr), ct1 < th, ht, nt))
    xr['curve_m1'] = cv
    xr['base_seen_attack_alarm'] = base_alarm
    xr['reproduced'] = True
    r['xgb'] = xr
audit.write('ALL PASS\n')
audit.close()
json.dump(out, open(os.path.join(R, 'nested_wn.json'), 'w'), indent=1)
