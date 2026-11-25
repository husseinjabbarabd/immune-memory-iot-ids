import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import study_audit as t

G, cls, att = t.G, t.cls, t.att


def prf(M, rows, cols):
    o = {}
    for k in cols:
        tp = M[k, k]
        ps = M[rows, k].sum()
        rs = M[k].sum()
        p = tp / ps if ps else 0.0
        r = tp / rs if rs else 0.0
        o[cls[k]] = [p, r, 2 * p * r / (p + r) if p + r else 0.0]
    return o


res = {'fwd16': prf(G, list(range(16)), list(range(16))),
       'att15a': prf(G, att, att),
       'att15b': prf(G, list(range(16)), att)}
mf = {k: float(np.mean([v[2] for v in d.values()])) for k, d in res.items()}
json.dump({'per_class': res, 'macro_f1': mf}, open(os.path.join(t.OUT, 'study_perclass.json'), 'w'), indent=1)
