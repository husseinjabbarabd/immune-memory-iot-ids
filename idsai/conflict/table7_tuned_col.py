import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def wilson(k, n, z=1.959963984540054):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


R = config.out('conflict')
d = json.load(open(os.path.join(R, 'conflict_decomp.json')))
out = {}
for g, v in d.items():
    if isinstance(v, dict) and 'xgb_tuned_rerun' in v and isinstance(v['xgb_tuned_rerun'], dict):
        x = v['xgb_tuned_rerun']
        n = v.get('n') or v.get('rows')
        out[g] = dict(acc=100 * x['acc'], errors=x['errors'], avoidable=x['avoidable'], n=n)
        if n and n < 1000:
            lo, hi = wilson(n - x['errors'], n)
            out[g]['wilson95'] = [100 * lo, 100 * hi]
json.dump(out, open(os.path.join(R, 'table7_tuned_col.json'), 'w'), indent=1)
