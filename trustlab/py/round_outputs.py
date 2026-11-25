import csv
import os
import re

import numpy as np
from scipy.io import loadmat

import config

wd = os.path.join(config.RES, 'nested')
closed = ['TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Bwd Pkt Len Max', 'Pkt Size Avg', 'Src Port', 'Bwd IAT Tot',
          'PSH Flag Cnt', 'Fwd Header Len', 'Dst Port', 'Bwd Byts/b Avg']
info = list(csv.DictReader(open(os.path.join(wd, 'rounds.csv'), encoding='utf-8')))
info += list(csv.DictReader(open(os.path.join(wd, 'joint.csv'), encoding='utf-8')))
info = {int(r['round']): r for r in info}


def g(o, f):
    return float(np.asarray(getattr(o, f)).ravel()[0])


rows = []
feat = {}
for r in range(1, 17):
    o = loadmat(os.path.join(wd, 'round_%02d.mat' % r), squeeze_me=True, struct_as_record=False)['o']
    assert int(g(o, 'round')) == r
    fs = [x.strip() for x in info[r]['feat'].split(';')]
    w = [int(x) for x in re.findall(r'\d+', info[r]['w'])]
    k = int(g(o, 'k'))
    assert k == len(fs) == len(w), r
    feat[r] = fs
    d = {'round': r, 'family': info[r]['family'], 'm': k,
         'features (weight)': '; '.join('%s (%d)' % (a, b) for a, b in zip(fs, w)),
         'shared with Sec. 4.4': sum(a in closed for a in fs), 'cells': int(g(o, 'cells')), 'AT': g(o, 'AT'),
         'nu': g(o, 'nu')}
    for j, op in [(1, 'SPEC'), (2, 'F2'), (3, 'R95')]:
        d['tau_' + op] = g(o, 'tau%d' % j)
        d['tauM_' + op] = g(o, 'tauM%d' % j)
        d['nuM_' + op] = g(o, 'nuM%d' % j)
    for l, rn in (('o', 'combined'), ('a', 'base'), ('c', 'control'), ('q', 'arm')):
        d['kno_%s_R95' % rn] = g(o, 'kno_%s3' % l)
    rows.append(d)

keys = list(rows[0])
with open(os.path.join(wd, 'round_outputs.csv'), 'w', newline='', encoding='utf-8') as f:
    wr = csv.DictWriter(f, keys)
    wr.writeheader()
    for d in rows:
        wr.writerow({k: ('%.10g' % v if isinstance(v, float) else v) for k, v in d.items()})

L = []
for c in ['m', 'shared with Sec. 4.4', 'cells', 'AT', 'nu'] + [p + '_' + op for op in ('SPEC', 'F2', 'R95') for p in ('tau', 'tauM', 'nuM')] + ['kno_%s_R95' % r for r in ('combined', 'base', 'control', 'arm')]:
    v = [d[c] for d in rows]
    L.append('%s: min %s, max %s' % (c, min(v), max(v)))
allf = [a for r in feat for a in feat[r]]
L.append('features in all 16 rounds: %s' % sorted({a for a in allf if allf.count(a) == 16}))
L.append('distinct features over 16 rounds: %d' % len(set(allf)))
open(os.path.join(wd, 'round_outputs.txt'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
