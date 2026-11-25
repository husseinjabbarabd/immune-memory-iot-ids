import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

OUT = config.out('fpr')
d = json.load(open(config.PROSPECTIVE))
T = ['0.005', '0.01', '0.02', '0.05']
fams = list(d['immune'])
oc = d['oneclass']


def row(h, f):
    im, xg = d['immune'][h][f], d['xgb'][h][f]
    r = {'immune': {'ok': im['attainable']}, 'xgb': {}}
    if im['attainable']:
        r['immune'].update(cal=100 * im['cv_fpr'], FPR=im['FPR'], ADR=im['ADR'], EUR=im['EUR'])
    else:
        r['immune']['cal_min'] = 100 * im['cv_fpr_min']
    x_ok = xg['cal_fpr'] <= float(f)
    r['xgb'] = {'ok': x_ok, 'cal': 100 * xg['cal_fpr'], 'FPR': xg['FPR'], 'ADR': xg['ADR'], 'EUR': xg['EUR']}
    for m in ('IF', 'OCSVM'):
        r[m] = {'ok': True, 'FPR': oc[m][f]['FPR'], 'ADR': oc[m][f]['ADR'][h], 'EUR': oc[m][f]['ADR'][h]}
    return r


res = {'per_round': {}, 'means_own_attainable': {}, 'means_shared': {}}
for f in T:
    R = {h: row(h, f) for h in fams}
    res['per_round'][f] = R
    shared = [h for h in fams if all(R[h][m]['ok'] for m in R[h])]
    for key, sel in (('means_own_attainable', None), ('means_shared', shared)):
        M = {}
        for m in ('immune', 'IF', 'OCSVM', 'xgb'):
            hs = [h for h in (sel if sel is not None else fams) if R[h][m]['ok']]
            M[m] = {'n_rounds': len(hs), 'rounds': hs if len(hs) < len(fams) else 'all'}
            for k in ('ADR', 'EUR', 'FPR', 'cal'):
                v = [R[h][m][k] for h in hs if k in R[h][m]]
                if v and len(v) == len(hs):
                    M[m][k] = float(np.mean(v))
        res[key][f] = M
res['attainable_rounds'] = {m: {f: sum(res['per_round'][f][h][m]['ok'] for h in fams) for f in T} for m in ('immune', 'xgb')}
json.dump(res, open(os.path.join(OUT, 'fpr_table.json'), 'w'), indent=1)
