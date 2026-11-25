import csv
import json
import os

import numpy as np
from scipy.io import loadmat

import config

wd = os.path.join(config.RES, 'portfree')
twins = list(config.TWINS)
T = json.load(open(os.path.join(wd, 'nested_table.json')))
fam = {d['round']: d['family'] for d in T['rounds']}
rl = [('full', 'o'), ('base', 'a'), ('control', 'c'), ('arm', 'q')]


def g(o, f):
    return float(np.asarray(getattr(o, f)).ravel()[0])


per = []
for r in range(1, 17):
    o = loadmat(os.path.join(wd, 'round_%02d.mat' % r), squeeze_me=True, struct_as_record=False)['o']
    d = {'round': r, 'family': fam[r], 'k': int(g(o, 'k')), 'cells': int(g(o, 'cells')), 'AT': g(o, 'AT'),
         'tau3': g(o, 'tau3'), 'tauM3': g(o, 'tauM3'), 'nu': g(o, 'nu'), 'nuM3': g(o, 'nuM3')}
    for n, l in rl:
        d['kno_' + n] = g(o, 'kno_%s3' % l)
    per.append(d)
s13 = [d for d in per if d['round'] <= 15 and d['family'] not in twins]
s15 = [d for d in per if d['round'] <= 15]
kno = {nm: {n: float(np.mean([d['kno_' + n] for d in ds])) for n, _ in rl} for nm, ds in (('i_13', s13), ('ii_15', s15))}
z = [x for x in csv.DictReader(open(os.path.join(wd, 'zd_original_protocol.csv'))) if x['cls'] not in twins]
assert len(z) == 13


def mz(k):
    return float(np.mean([float(x[k]) for x in z]))


sub = {n: {'det': mz('det_%s3' % l), 'far': mz('far_%s3' % l), 'kno': mz('kno_%s3' % l)} for n, l in rl}
P = json.load(open(os.path.join(config.RES, 'nested', 'nested_table.json')))['aggregates']
A = T['aggregates']
diff = {nm: {n: 100 * (A[nm]['det_' + n] - P[nm]['det_' + n]) for n, _ in rl} for nm in ('i_13', 'ii_15', 'iii_15_joint')}
far = {nm: {n: 100 * (A[nm]['far_' + n] - P[nm]['far_' + n]) for n, _ in rl} for nm in ('i_13', 'ii_15', 'iii_15_joint')}
json.dump({'rounds': per, 'kno_means': kno, 'submitted_portfree_13': sub, 'det_minus_ports_pp': diff,
           'far_minus_ports_pp': far, 'k_all': sorted({d['k'] for d in per})},
          open(os.path.join(wd, 'portfree_nested_extra.json'), 'w'), indent=1)


def pc(x):
    return '%.2f' % (100 * x)


L = ['k per round: %s' % [d['k'] for d in per], 'cells per round: %s' % [d['cells'] for d in per]]
for nm, v in kno.items():
    L.append('known-attack recall %s: ' % nm + '; '.join('%s %s' % (n, pc(v[n])) for n, _ in rl))
L.append('submitted protocol without ports, 13: ' + '; '.join('%s %s / %.3f / kno %s' % (n, pc(v['det']), 100 * v['far'], pc(v['kno'])) for n, v in sub.items()))
for nm in diff:
    L.append('det without minus with ports %s: ' % nm + '; '.join('%s %+.2f' % (n, diff[nm][n]) for n, _ in rl))
    L.append('FAR without minus with ports %s: ' % nm + '; '.join('%s %+.3f' % (n, far[nm][n]) for n, _ in rl))
open(os.path.join(wd, 'portfree_nested_extra.txt'), 'w').write('\n'.join(L) + '\n')
