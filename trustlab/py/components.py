import csv
import os

import config

D = os.path.join(config.RES, 'components')
TWINS = config.TWINS
COLS = ['auc', 'tau', 'nFwd', 'P1_Rec', 'P1_Spec', 'P1_F2', 'fwd_Acc', 'fwd_MF1', 'e2e_Acc', 'e2e_MF1']


def rd(f):
    return list(csv.DictReader(open(os.path.join(D, f))))


def mean(v):
    return sum(v) / len(v)


L = []
F = rd('select_features.csv')
mi = [r['feature'] for r in F if r['mi_top10'] == '1']
cl = [r['feature'] for r in F if r['clonalg'] == '1']
L += ['features', '  MI top 10: ' + '; '.join(mi), '  CLONALG: ' + '; '.join(cl),
      '  shared: %d (%s)' % (len(set(mi) & set(cl)), '; '.join(f for f in mi if f in cl)), '']

for name, rows, key in (('memory_results.csv', rd('memory_results.csv'), 'arm'),
                        ('select_results.csv', rd('select_results.csv'), 'arm'),
                        ('stack_results.csv', rd('stack_results.csv'), None)):
    L.append(name)
    L.append('  %-42s %-8s ' % ('arm', 'op') + ' '.join('%11s' % c for c in COLS))
    for r in rows:
        L.append('  %-42s %-8s ' % ((r[key] if key else 'two-phase stack')[:42], r['opPoint'])
                 + ' '.join('%11.6g' % float(r[c]) for c in COLS))
    L.append('')

O = rd('open_nested.csv')
J = rd('open_nested_joint.csv')
rules = [k[4:] for k in O[0] if k.startswith('det_')]
single = [r for r in O if '+' not in r['family']]
r13 = [r for r in single if r['family'] not in TWINS]
jr = [r for r in O if '+' in r['family']][0]
jd = {(r['class'], r['rule']): float(r['det']) for r in J}
L.append('open_nested.csv (16 nested rounds); means (i) 13 families, (ii) 15 single-family rounds, '
         '(iii) 15 families with DoS and Slowloris from the joint round (detection only)')
L.append('  %-14s %9s %9s %9s %9s %9s %9s' % ('rule', 'det(i)', 'det(ii)', 'det(iii)', 'far(i)', 'kno(i)', 'vfar(i)'))
for rr in rules:
    d3 = [float(r['det_' + rr]) for r in r13] + [jd[(t, rr)] for t in TWINS]
    L.append('  %-14s %9.5f %9.5f %9.5f %9.5f %9.5f %9.5f' % (
        rr, mean([float(r['det_' + rr]) for r in r13]), mean([float(r['det_' + rr]) for r in single]), mean(d3),
        mean([float(r['far_' + rr]) for r in r13]), mean([float(r['kno_' + rr]) for r in r13]),
        mean([float(r['vfar_' + rr]) for r in r13])))
L.append('  joint round %s: ' % jr['family'] + '; '.join(
    '%s %s %.5f' % (t, rr, jd[(t, rr)]) for rr in ('combined3', 'nsa3', 'n09_3') for t in TWINS))
L.append('')
L.append('  per round, OP-R95 matched (det combined3 / nsa3 / n09_3; detectors / max)')
for r in O:
    L.append('  %-16s %8.5f %8.5f %8.5f   %s/%s' % (r['family'], float(r['det_combined3']), float(r['det_nsa3']),
                                                  float(r['det_n09_3']), r['detectors'], r['max_detectors']))
L.append('  own rules, per round (det / far: nsa_native, n09_native)')
for r in O:
    L.append('  %-16s %8.5f %8.5f   %8.5f %8.5f' % (r['family'], float(r['det_nsa_native']), float(r['far_nsa_native']),
                                                  float(r['det_n09_native']), float(r['far_n09_native'])))
open(os.path.join(D, 'summary.txt'), 'w').write('\n'.join(L) + '\n')
