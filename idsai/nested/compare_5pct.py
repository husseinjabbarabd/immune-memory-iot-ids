import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

R = config.out('nested')
nest = json.load(open(os.path.join(R, 'nested_fpr.json')))
rows = {r['family']: (float(r['ADR']), float(r['EUR'])) for r in csv.DictReader(open(config.TABLE10, encoding='utf-8'))}
assert len(rows) == 10, rows
mean = [round(np.mean([v[i] for v in rows.values()]) + 1e-12, 2) for i in (0, 1)]
assert mean == [88.95, 62.87], mean

fam = list(rows)
att = [f for f in fam if nest[f]['immune']['0.05'].get('attainable')]
res = dict(families=fam, attainable=att, original_mean=dict(ADR=mean[0], EUR=mean[1]), per_family={})
for f in fam:
    n = nest[f]['immune']['0.05']
    res['per_family'][f] = dict(original=dict(ADR=rows[f][0], EUR=rows[f][1]),
                                nested=dict(ADR=n.get('ADR'), EUR=n.get('EUR'), FPR=n.get('FPR'), cal_fpr=n.get('cal_fpr')),
                                diff=dict(ADR=n['ADR'] - rows[f][0], EUR=n['EUR'] - rows[f][1]) if n.get('attainable') else None)
for k, i in (('ADR', 0), ('EUR', 1)):
    nm = float(np.mean([nest[f]['immune']['0.05'][k] for f in att]))
    om = float(np.mean([rows[f][i] for f in att]))
    res['over_attainable_' + k] = dict(n=len(att), nested=nm, original_same_families=om, diff=nm - om)
    if len(att) == 10:
        res['all10_' + k] = dict(nested=nm, original=mean[i], diff=nm - mean[i])
res['nested_mean_FPR'] = float(np.mean([nest[f]['immune']['0.05']['FPR'] for f in att]))
json.dump(res, open(os.path.join(R, 'compare_5pct.json'), 'w'), indent=1)
