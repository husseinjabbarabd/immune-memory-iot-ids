import json
import os
import sys

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.chdir(config.WORK)
s = np.load('split.npz', allow_pickle=True)
Xtr, ytr, Xte, yte = s['Xtr'], s['ytr'], s['Xte'], s['yte']
classes = [str(c) for c in s['classes']]
K = 11
N = classes.index('Normal')
T = [0.005, 0.01, 0.02, 0.05]
FG = np.round(np.arange(1, 201) * 0.0005, 4)
ref = json.load(open(config.PROSPECTIVE))['oneclass']
out = {}
X1, X2 = Xtr.copy(), Xte.copy()
for X in (X1, X2):
    X[:, 0] = np.arcsinh(1e6 * X[:, 0])
rng = np.random.default_rng(21)
nidx = np.where(ytr == N)[0]
rng.shuffle(nidx)
fi, ca = nidx[:320000], nidx[320000:]
assert (ytr[fi] == N).all() and (ytr[ca] == N).all()
sc = StandardScaler().fit(X1[fi])
fam = [h for h in range(K) if h != N]
for name, mk, sub in [('IF', lambda: IsolationForest(n_estimators=200, max_samples=4096, random_state=21, n_jobs=2), fi),
                      ('OCSVM', lambda: OneClassSVM(kernel='rbf', gamma='scale', nu=0.02), fi[:20000])]:
    m = mk().fit(sc.transform(X1[sub]))
    scal = -m.score_samples(sc.transform(X1[ca]))
    ste = -m.score_samples(sc.transform(X2))
    r = {}
    for f in T:
        th = np.quantile(scal, 1 - f)
        fl = ste > th
        fpr, adr = 100 * float(fl[yte == N].mean()), {classes[h]: 100 * float(fl[yte == h].mean()) for h in fam}
        assert abs(fpr - ref[name][str(f)]['FPR']) < 1e-9 and all(abs(adr[k] - ref[name][str(f)]['ADR'][k]) < 1e-9
                                                                  for k in adr), name
        r[str(f)] = dict(cal_fpr=100 * float((scal > th).mean()), FPR=fpr, ADR=adr)
    pts = []
    for f in FG:
        th = np.quantile(scal, 1 - f)
        fl = ste > th
        pts.append([100 * float((scal > th).mean()), 100 * float(fl[yte == N].mean())] + [100 * float(fl[yte == h].mean()) for h in fam])
    r['curve'] = dict(columns=['cal_fpr', 'test_fpr'] + [classes[h] for h in fam], points=pts)
    out[name] = r
json.dump(out, open(os.path.join(config.out('nested'), 'curves_oneclass.json'), 'w'), indent=1)
