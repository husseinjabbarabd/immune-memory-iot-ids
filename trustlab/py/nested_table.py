import csv
import json
import os
import sys

import numpy as np
from scipy.io import loadmat

import config

tag = sys.argv[1]
wd = os.path.join(config.RES, tag)
info = list(csv.DictReader(open(os.path.join(wd, 'rounds.csv'), encoding='utf-8')))
info += list(csv.DictReader(open(os.path.join(wd, 'joint.csv'), encoding='utf-8')))
fam = {int(r['round']): r['family'] for r in info}
assert fam[16] == 'DoS+Slowloris'
rules = [('full', 'o'), ('base', 'a'), ('control', 'c'), ('arm', 'q')]
twins = list(config.TWINS)


def get(o, f):
    return float(np.asarray(getattr(o, f)).ravel()[0])


def load(r):
    return loadmat(os.path.join(wd, 'round_%02d.mat' % r), squeeze_me=True, struct_as_record=False)['o']


rows = []
for r in range(1, 17):
    o = load(r)
    assert int(get(o, 'round')) == r
    d = {'round': r, 'family': fam[r], 'k': int(get(o, 'k')), 'n': int(get(o, 'nX')),
         'tieVal': int(get(o, 'tieVal')), 'tieTest': int(get(o, 'tieTest')),
         'tau': get(o, 'tau3'), 'tauM': get(o, 'tauM3'), 'nu': get(o, 'nu'), 'nuM': get(o, 'nuM3')}
    for rn, l in rules:
        d['det_' + rn] = get(o, 'det_%s3' % l)
        d['far_' + rn] = get(o, 'far_%s3' % l)
        d['vfar_' + rn] = get(o, 'vfar_%s3' % l)
    if r == 16:
        for z in twins:
            for rn, f2 in [('full', 'det'), ('base', 'detA'), ('control', 'detC'), ('arm', 'detQ')]:
                d['det_%s_%s' % (rn, z)] = get(o, '%s_%s_3' % (f2, z))
    rows.append(d)

R = {d['family']: d for d in rows}
single = [d for d in rows if d['round'] <= 15]


def agg(ds, src=None):
    out = {}
    for rn, _ in rules:
        for m in ('det', 'far', 'vfar'):
            v = []
            for d in ds:
                if src and d['family'] in twins and m == 'det':
                    v.append(src['det_%s_%s' % (rn, d['family'])])
                elif src and d['family'] in twins:
                    v.append(src['%s_%s' % (m, rn)])
                else:
                    v.append(d['%s_%s' % (m, rn)])
            out['%s_%s' % (m, rn)] = float(np.mean(v))
    out['n_rounds'] = len(ds)
    return out


A = {}
s13 = [d for d in single if d['family'] not in twins]
A['i_13'] = agg(s13)
A['ii_15'] = agg(single)
A['iii_15_joint'] = agg(single, R['DoS+Slowloris'])


def cmp(ds, a, b, src=None):
    bt = eq = wo = 0
    for d in ds:
        if src and d['family'] in twins:
            x, y = src['det_%s_%s' % (a, d['family'])], src['det_%s_%s' % (b, d['family'])]
        else:
            x, y = d['det_' + a], d['det_' + b]
        bt += x > y
        eq += x == y
        wo += x < y
    return [bt, eq, wo]


C = {}
for nm, ds, src in [('i_13', s13, None), ('ii_15', single, None), ('iii_15_joint', single, R['DoS+Slowloris'])]:
    C[nm] = {'full_vs_control': cmp(ds, 'full', 'control', src), 'full_vs_arm': cmp(ds, 'full', 'arm', src),
             'full_vs_base': cmp(ds, 'full', 'base', src)}
ties = {'val': sum(d['tieVal'] for d in rows), 'test': sum(d['tieTest'] for d in rows),
        'rounds_with_test_ties': [d['family'] for d in rows if d['tieTest'] > 0]}

OPS = {}
o16 = load(16)
for j, opn in [(1, 'OP-SPEC'), (2, 'OP-F2'), (3, 'OP-R95')]:
    per = []
    for r in range(1, 16):
        o = load(r)
        per.append({'family': fam[r], **{'%s_%s' % (m, rn): get(o, '%s_%s%d' % (m, l, j)) for rn, l in rules for m in ('det', 'far', 'vfar')}})
    p13 = [d for d in per if d['family'] not in twins]
    src = {'%s_%s' % (m, rn): get(o16, '%s_%s%d' % (m, l, j)) for rn, l in rules for m in ('far', 'vfar')}
    src.update({'det_%s_%s' % (rn, z): get(o16, '%s_%s_%d' % (f2, z, j)) for z in twins
                for rn, f2 in [('full', 'det'), ('base', 'detA'), ('control', 'detC'), ('arm', 'detQ')]})
    OPS.setdefault('iii', {})[opn] = {'mean': agg(per, src), 'counts': {'full_vs_control': cmp(per, 'full', 'control', src),
                                                                        'full_vs_arm': cmp(per, 'full', 'arm', src)}}
    OPS[opn] = {'i_13': {k: float(np.mean([d[k] for d in p13])) for k in p13[0] if k != 'family'},
                'ii_15': {k: float(np.mean([d[k] for d in per])) for k in per[0] if k != 'family'},
                'counts_13': {'full_vs_control': cmp(p13, 'full', 'control'), 'full_vs_arm': cmp(p13, 'full', 'arm')},
                'counts_15': {'full_vs_control': cmp(per, 'full', 'control'), 'full_vs_arm': cmp(per, 'full', 'arm')}}

def mn(rs, k):
    return float(np.mean([float(x[k]) for x in rs]))


SUB = None
if tag == 'nested':
    s2 = [x for x in csv.DictReader(open(os.path.join(config.REFZD, 'ZD_S02', 'ZD_S02_base_vs_self_novelty.csv'))) if x['twin'] == '0']
    s3 = [x for x in csv.DictReader(open(os.path.join(config.REFZD, 'ZD_S03', 'ZD_S03_equal_far.csv'))) if x['twin'] == '0']
    assert [x['class'] for x in s2] == [x['class'] for x in s3]
    SUB = {'n': len(s2), 'det_full': mn(s2, 'detOR_3'), 'far_full': mn(s2, 'farOR_3'), 'vfar_full': mn(s2, 'farValOR_3'),
           'det_base': mn(s2, 'detBase_3'), 'far_base': mn(s2, 'farBase_3'),
           'det_control': mn(s3, 'detM_3'), 'far_control': mn(s3, 'farM_3'), 'vfar_control': mn(s3, 'farValM_3'),
           'count_full_vs_control': cmp([{'det_full': float(a['detOR_3']), 'det_control': float(b['detM_3']), 'family': a['class']}
                                         for a, b in zip(s2, s3)], 'full', 'control')}
json.dump({'rounds': rows, 'aggregates': A, 'counts': C, 'ties': ties, 'ops': OPS, 'submitted_r95_13': SUB},
          open(os.path.join(wd, 'nested_table.json'), 'w'), indent=1)


def pc(x):
    return '%.2f' % (100 * x)


def pf(x):
    return '%.3f' % (100 * x)


def line(a):
    return '; '.join('%s %s / %s / %s' % (rn, pc(a['det_' + rn]), pf(a['far_' + rn]), pf(a['vfar_' + rn])) for rn, _ in rules)


L = ['round | family | k | n test | ' + ' | '.join('%s det / FAR / vFAR' % rn for rn, _ in rules) + ' | ties val/test']
for d in rows:
    L.append('%d | %s | %d | %d | ' % (d['round'], d['family'], d['k'], d['n']) +
             ' | '.join('%s / %s / %s' % (pc(d['det_' + rn]), pf(d['far_' + rn]), pf(d['vfar_' + rn])) for rn, _ in rules) +
             ' | %d / %d' % (d['tieVal'], d['tieTest']))
    if d['round'] == 16:
        for z in twins:
            L.append('16 | %s (joint) | | | ' % z + ' | '.join(pc(d['det_%s_%s' % (rn, z)]) for rn, _ in rules))
L.append('')
for nm, a in A.items():
    L.append('%s: %s' % (nm, line(a)))
for nm, c in C.items():
    L.append('%s counts (higher/equal/lower): %s' % (nm, json.dumps(c)))
L.append('ties (dA = dB = 0): validation %d, test %d, test rounds %s' % (ties['val'], ties['test'], ties['rounds_with_test_ties']))
for opn in ('OP-SPEC', 'OP-F2', 'OP-R95'):
    v = OPS[opn]
    for nm in ('i_13', 'ii_15'):
        L.append('%s %s: %s' % (opn, nm, line(v[nm])))
    L.append('%s counts 13: %s; 15: %s' % (opn, json.dumps(v['counts_13']), json.dumps(v['counts_15'])))
    x = OPS['iii'][opn]
    L.append('%s iii_15_joint: %s' % (opn, line(x['mean'])))
    L.append('%s counts iii: %s' % (opn, json.dumps(x['counts'])))
if SUB:
    L.append('submitted OP-R95, %d families: ' % SUB['n'] + json.dumps({k: (round(100 * x, 4) if isinstance(x, float) else x) for k, x in SUB.items()}))
open(os.path.join(wd, 'nested_table.txt'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
