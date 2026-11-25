import json
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.chdir(config.WORK)
ref = json.load(open(config.PROSPECTIVE))['xgb']
s = np.load('split.npz', allow_pickle=True)
Xtr, ytr, Xte, yte = s['Xtr'], s['ytr'], s['Xte'], s['yte']
classes = [str(c) for c in s['classes']]
K = 11
N = classes.index('Normal')
T = [0.005, 0.01, 0.02, 0.05]
out = {}
for name in [c for c in classes if c != 'Normal']:
    h = classes.index(name)
    keep = ytr != h
    X, y = Xtr[keep], ytr[keep]
    kc = [k for k in range(K) if k != h]
    m_ = np.full(K, -1)
    m_[kc] = np.arange(10)
    yy = m_[y]
    Nn = m_[N]
    Xf, Xc, yf, yc = train_test_split(X, yy, test_size=0.2, stratify=yy, random_state=21)
    m = XGBClassifier(eval_metric='mlogloss', n_jobs=2, colsample_bytree=0.75, gamma=1.5, max_depth=6,
                      min_child_weight=1, subsample=1, learning_rate=0.3, n_estimators=100,
                      tree_method='exact', random_state=0).fit(Xf, yf)
    Pc = m.predict_proba(Xc[yc == Nn])
    Pt = m.predict_proba(Xte)
    pc, mc = Pc.argmax(1), Pc.max(1)
    pt, mt = Pt.argmax(1), Pt.max(1)
    att = pt != Nn
    n = len(mc)
    u = int((pc != Nn).sum())
    ms = np.sort(mc[pc == Nn])
    r = dict(n=n, u=u)
    ok = True
    for f in T:
        need = f - (pc != Nn).mean()
        if need <= 0:
            th = 0.0
            k = 0
        else:
            k = int(round(need * n))
            th = float(ms[k - 1]) + 1e-12 if k > 0 else 0.0
        fl = mt < th
        v = dict(EUR=100 * float(fl[yte == h].mean()), ADR=100 * float((fl | att)[yte == h].mean()),
                 FPR=100 * float((fl | att)[yte == N].mean()), cal_fpr=float(((mc < th) | (pc != Nn)).mean()))
        same = all(abs(v[q] - ref[name][str(f)][q]) < 1e-9 for q in v)
        ok = ok and same
        e = dict(k=k, flagged=int((mc < th).sum()), same=same, **v)
        if k > 0:
            vk = ms[k - 1]
            e.update(v_k=float(vk), tied=int((ms == vk).sum()), tied_in_k=int((ms[:k] == vk).sum()))
        r[str(f)] = e
    r['reproduced'] = ok
    out[name] = r
json.dump(out, open(os.path.join(config.out('fpr'), 'xgb_ties.json'), 'w'), indent=1)
