import csv
import os

import numpy as np
from scipy.io import loadmat

import config

OUT = os.path.join(config.RES, 'common_far')
NAMES = config.CLASSES
BEN = 2
TWINS = config.TWINS
ALPHA = 0.005
TARGET = 0.01169
GRID = np.round(np.arange(1, 101) / 1000, 3)
VER = {
    'port-using': dict(wd=os.path.join(config.RES, 'pipeline'), csv='zd_original_protocol.csv', taus='test_rows.mat'),
    'port-free': dict(wd=os.path.join(config.RES, 'portfree'), csv='zd_original_protocol.csv', taus='closed_state.mat'),
}
DET = ('full rule', 'control', 'arm alone')


def roundk(x):
    return int(np.floor(x + 0.5))


def match_k(s, K):
    n = s.size
    if K < 1 or n == 0:
        return np.inf
    sd = np.sort(s)[::-1]
    v = sd[min(K, n) - 1]
    if np.sum(s >= v) <= K:
        return v
    hi = sd[sd > v]
    return hi.min() if hi.size else np.inf


def match(s, far):
    return match_k(s, roundk(far * s.size))


def full_tau(sv, qv, nu, far):
    base = qv >= nu
    k = roundk(far * sv.size) - int(base.sum())
    if k < 0:
        return None
    return match_k(sv[~base], k)


def quantile(x, p):
    x = np.sort(x)
    n = x.size
    r = (100.0 * p / 100) * n
    k = int(np.floor(r + 0.5))
    r = r - k
    return (0.5 + r) * x[k] + (0.5 - r) * x[k - 1]


def sq(f, F):
    S = loadmat(f, variable_names=['dA', 'dB'])
    dA = S['dA'].ravel().astype(float)
    dB = S['dB'].ravel().astype(float)
    sA, sB = np.sqrt(dA), np.sqrt(dB)
    with np.errstate(invalid='ignore', divide='ignore'):
        s = sB / (sA + sB)
    s[sA + sB == 0] = 0.5
    return s, sB / (1023 * np.sqrt(F)), dB


def feat(dB, qx):
    i = np.flatnonzero(dB > 0)[:1000]
    for F in range(1, 120):
        if np.array_equal((np.sqrt(dB[i]) / (1023 * np.sqrt(F))).astype(np.float32), qx[i]):
            return F
    raise SystemExit('feature count not found')


def g15(v):
    return float('%.15g' % v)


def same(a, b):
    return g15(a) == g15(float(b))


def check(ver, x, r, s, q, sv, qv, nu, taus, isX, isB, isK, bad):
    def put(name, val):
        if name in r and not same(val, r[name]):
            bad.append('%s %s %s: %r vs %r' % (ver, NAMES[x - 1], name, val, r[name]))
    put('nu', nu)
    for o in range(3):
        t = taus[o]
        farV = np.mean((sv >= t) | (qv >= nu))
        tM = match(sv, farV)
        nM = match(qv, farV)
        R = {'a': s >= t, 'o': (s >= t) | (q >= nu), 'c': s >= tM, 'q': q >= nM}
        j = o + 1
        if ver == 'port-free':
            put('tauM%d' % j, tM)
            put('nuM%d' % j, nM)
            for k, v in R.items():
                put('det_%s%d' % (k, j), v[isX].mean())
                put('far_%s%d' % (k, j), v[isB].mean())
                put('kno_%s%d' % (k, j), v[isK].mean())
        else:
            put('tauM%d' % j, tM)
            put('det%d' % j, R['a'][isX].mean())
            put('far%d' % j, R['a'][isB].mean())
            put('detOR%d' % j, R['o'][isX].mean())
            put('farOR%d' % j, R['o'][isB].mean())
            put('detM%d' % j, R['c'][isX].mean())
            put('farM%d' % j, R['c'][isB].mean())


def rules(sv, qv, nu, far):
    t = full_tau(sv, qv, nu, far)
    return {'full rule': None if t is None else (t, nu), 'control': (match(sv, far), np.inf), 'arm alone': (np.inf, match(qv, far))}


def apply(th, s, q):
    t, n = th
    return (s >= t) | (q >= n)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, curve, bad, info = [], {}, [], []
    for ver, v in VER.items():
        ref = {r['cls']: r for r in csv.DictReader(open(os.path.join(v['wd'], v['csv']), encoding='utf-8'))}
        yTe = loadmat(os.path.join(v['wd'], 'test_rows.mat'), variable_names=['yTe'])['yTe'].ravel().astype(int)
        taus = loadmat(os.path.join(v['wd'], v['taus']), variable_names=['taus'])['taus'].ravel().astype(float)
        F = None
        for x in range(1, 17):
            if x == BEN:
                continue
            fam = NAMES[x - 1]
            Z = loadmat(os.path.join(v['wd'], 'zd_%02d.mat' % x), variable_names=['sx', 'qx', 'nu'])
            sx, qx, nu = Z['sx'].ravel(), Z['qx'].ravel(), float(Z['nu'].ravel()[0])
            d = os.path.join(v['wd'], 'zd%02d' % x)
            if F is None:
                dB = loadmat(os.path.join(d, 'score_test.mat'), variable_names=['dB'])['dB'].ravel().astype(float)
                F = feat(dB, qx)
            s, q, _ = sq(os.path.join(d, 'score_test.mat'), F)
            sv, qv, _ = sq(os.path.join(d, 'score_val.mat'), F)
            assert s.size == yTe.size == sx.size
            if not (np.array_equal(s.astype(np.float32), sx) and np.array_equal(q.astype(np.float32), qx)):
                bad.append('%s %s: test scores differ from sx/qx' % (ver, fam))
            nq = quantile(qv, 1 - ALPHA)
            info.append('%s %-15s F=%d nVal=%d nTest=%d nu saved %.17g recomputed %.17g' % (ver, fam, F, sv.size, s.size, nu, nq))
            if nq != nu:
                bad.append('%s %s: nu %r vs recomputed %r' % (ver, fam, nu, nq))
            isX = yTe == x
            isB = yTe == BEN
            isK = ~isX & ~isB
            check(ver, x, ref[fam], s, q, sv, qv, nu, taus, isX, isB, isK, bad)
            if fam in TWINS:
                continue
            for k, th in rules(sv, qv, nu, TARGET).items():
                if th is None:
                    rows.append(dict(version=ver, family=fam, detector=k, attainable=0))
                    continue
                f = apply(th, s, q)
                fv = apply(th, sv, qv)
                rows.append(dict(version=ver, family=fam, detector=k, attainable=1, tau=th[0], nu=th[1], val_far=fv.mean(),
                                 test_far=f[isB].mean(), detection=f[isX].mean(), known=f[isK].mean()))
            for gv in GRID:
                for k, th in rules(sv, qv, nu, gv).items():
                    c = curve.setdefault((ver, k, gv), [])
                    if th is None:
                        c.append(None)
                    else:
                        f = apply(th, s, q)
                        c.append((f[isB].mean(), f[isX].mean()))
    open(os.path.join(OUT, 'checks.txt'), 'w').write('\n'.join(info + ['mismatches: %d' % len(bad)] + bad) + '\n')
    if bad:
        raise SystemExit('checks failed: %d mismatches, see checks.txt' % len(bad))

    keys = ['version', 'family', 'detector', 'attainable', 'tau', 'nu', 'val_far', 'test_far', 'detection', 'known']
    with open(os.path.join(OUT, 'rounds.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ('%.15g' % r[k] if isinstance(r.get(k), float) else r.get(k, '')) for k in keys})

    lines = ['target validation false-alarm rate %.5f; 13 families (15 rounds minus %s)' % (TARGET, ', '.join(TWINS)), '']
    lines.append('%-11s %-10s %3s %10s %10s %10s %10s' % ('version', 'detector', 'n', 'detection', 'test_FAR', 'val_FAR', 'known'))
    by = {}
    for r in rows:
        by.setdefault((r['version'], r['detector']), []).append(r)
    for ver in VER:
        for k in DET:
            R = [r for r in by[(ver, k)] if r['attainable']]
            lines.append('%-11s %-10s %3d %10.4f %10.5f %10.5f %10.4f' % (ver, k, len(R), 100 * np.mean([r['detection'] for r in R]),
                         100 * np.mean([r['test_far'] for r in R]), 100 * np.mean([r['val_far'] for r in R]), 100 * np.mean([r['known'] for r in R])))
            na = [r['family'] for r in by[(ver, k)] if not r['attainable']]
            if na:
                lines.append('    not attainable: ' + ', '.join(na))
    lines.append('')
    for ver in VER:
        d = {k: {r['family']: r for r in by[(ver, k)]} for k in DET}
        for other in ('control', 'arm alone'):
            fams = [f for f in d['full rule'] if d['full rule'][f]['attainable']]
            wins = sum(d['full rule'][f]['detection'] > d[other][f]['detection'] for f in fams)
            ties = sum(d['full rule'][f]['detection'] == d[other][f]['detection'] for f in fams)
            lines.append('%s: full rule detects more than %s in %d of %d rounds (ties %d)' % (ver, other, wins, len(fams), ties))
    lines.append('')

    pts = {}
    for (ver, k, gv), c in curve.items():
        if all(p is not None for p in c):
            pts.setdefault((ver, k), []).append((gv, np.mean([p[0] for p in c]), np.mean([p[1] for p in c])))
    lo = max(min(p[1] for p in P) for P in pts.values())
    hi = min(max(p[1] for p in P) for P in pts.values())
    lines.append('shared mean test false-alarm range: %.5f%% to %.5f%%' % (100 * lo, 100 * hi))
    for key, P in sorted(pts.items()):
        lines.append('%s %s: %d grid points kept, mean test FAR %.5f%% to %.5f%%' % (key[0], key[1], len(P), 100 * min(p[1] for p in P), 100 * max(p[1] for p in P)))
    with open(os.path.join(OUT, 'curves.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['version', 'detector', 'target_val_far', 'mean_test_far', 'mean_detection', 'in_shared_range'])
        for (ver, k), P in sorted(pts.items()):
            for gv, x, y in P:
                w.writerow([ver, k, '%.3f' % gv, '%.15g' % x, '%.15g' % y, int(lo <= x <= hi)])
    open(os.path.join(OUT, 'summary.txt'), 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
