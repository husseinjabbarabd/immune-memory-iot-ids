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
Xtr, ytr, Xte, yte = s['Xtr'].copy(), s['ytr'], s['Xte'].copy(), s['yte']
cl = [str(c) for c in s['classes']]
N = cl.index('Normal')
for X in (Xtr, Xte):
    X[:, 0] = np.arcsinh(1e6 * X[:, 0])
rng = np.random.default_rng(21)
nidx = np.where(ytr == N)[0]
rng.shuffle(nidx)
fit_i, cal_i = nidx[:320000], nidx[320000:]
sc = StandardScaler().fit(Xtr[fit_i])


def Z(X):
    return sc.transform(X)


targets = [0.0141, 0.0202]
out = {}
models = {'Isolation Forest': lambda: IsolationForest(n_estimators=200, max_samples=4096, random_state=21, n_jobs=2),
          'One-Class SVM (RBF)': lambda: OneClassSVM(kernel='rbf', gamma='scale', nu=0.02)}
for name, mk in models.items():
    m = mk()
    fi = fit_i if name.startswith('Iso') else fit_i[:20000]
    m.fit(Z(Xtr[fi]))
    sc_cal = -m.score_samples(Z(Xtr[cal_i]))
    sc_te = -m.score_samples(Z(Xte))
    res = {}
    for tf in targets:
        th = np.quantile(sc_cal, 1 - tf)
        flag = sc_te > th
        r = {'normal_fpr_test': float(flag[yte == N].mean())}
        r['per_attack_ADR'] = {cl[c]: float(100 * flag[yte == c].mean()) for c in range(11) if c != N}
        r['mean_ADR'] = float(np.mean(list(r['per_attack_ADR'].values())))
        res[f'FPR_{tf}'] = r
    out[name] = res
json.dump(out, open(os.path.join(config.out('base'), 'zd_baselines.json'), 'w'), indent=1)
