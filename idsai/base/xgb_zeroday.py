import json
import os
import sys

import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.chdir(config.WORK)
s = np.load('split.npz', allow_pickle=True)
Xtr, ytr, Xte, yte = s['Xtr'], s['ytr'], s['Xte'], s['yte']
cl = [str(c) for c in s['classes']]
N = cl.index('Normal')
targets = [0.0141, 0.0202]
out = {}
for h in [c for c in range(11) if c != N]:
    keep = ytr != h
    X, y = Xtr[keep], ytr[keep]
    kept = [c for c in range(11) if c != h]
    rm = {c: i for i, c in enumerate(kept)}
    yy = np.array([rm[c] for c in y])
    Nn = rm[N]
    Xf, Xc, yf, yc = train_test_split(X, yy, test_size=0.2, stratify=yy, random_state=21)
    m = XGBClassifier(eval_metric='mlogloss', n_jobs=2, colsample_bytree=0.75, gamma=1.5, max_depth=6, min_child_weight=1,
                      subsample=1, learning_rate=0.3, n_estimators=100, tree_method='exact', random_state=0)
    m.fit(Xf, yf)
    Pc = m.predict_proba(Xc[yc == Nn])
    Pt = m.predict_proba(Xte)
    pc, mc = Pc.argmax(1), Pc.max(1)
    pt, mt = Pt.argmax(1), Pt.max(1)
    att_t = pt != Nn
    r = {'cal_attack_share_noflag': float((pc != Nn).mean())}
    nt = yte == N
    ht = yte == h
    r['noflag'] = {'ADR': float(100 * att_t[ht].mean()), 'normal_fpr_test': float(100 * att_t[nt].mean())}
    for tf in targets:
        need = tf - (pc != Nn).mean()
        if need <= 0:
            th = 0.0
        else:
            mnorm = np.sort(mc[pc == Nn])
            k = int(round(need * len(mc)))
            th = float(mnorm[k - 1]) + 1e-12 if k > 0 else 0.0
        flag = mt < th
        eur = flag[ht].mean()
        known = ((~flag) & att_t)[ht].mean()
        r[f'FPR_{tf}'] = {'t': th, 'cal_fpr': float(((mc < th) | (pc != Nn)).mean()),
                          'EUR': float(100 * eur), 'Known': float(100 * known), 'ADR': float(100 * (eur + known)),
                          'normal_fpr_test': float(100 * (flag | att_t)[nt].mean()),
                          'absorbed_into': {cl[kept[c]]: int(((~flag) & (pt == c))[ht].sum()) for c in range(10)
                                            if c != Nn and ((~flag) & (pt == c))[ht].sum() > 0}}
    out[cl[h]] = r
json.dump(out, open(os.path.join(config.out('base'), 'xgb_zeroday.json'), 'w'), indent=1)
