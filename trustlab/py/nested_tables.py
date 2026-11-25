import csv
import os

import config

wd = os.path.join(config.RES, 'nested')
ro = list(csv.DictReader(open(os.path.join(wd, 'round_outputs.csv'), encoding='utf-8')))
bt = list(csv.DictReader(open(os.path.join(wd, 'boot.csv'), encoding='utf-8')))


def f3(x):
    return '%.3f' % float(x)


def f4(x):
    return '%.4f' % float(x)


def pc(x):
    return '%.1f' % (100 * float(x))


L = ['Table 11d', 'family | cells | AT | tau | tau_M | nu | nu_M | known-attack recall: combined / base / control / arm']
for d in ro:
    L.append(' | '.join([d['family'].replace('DoS+Slowloris', 'DoS + Slowloris, together'), '{:,}'.format(int(d['cells'])),
                         f3(d['AT']), f3(d['tau_R95']), f3(d['tauM_R95']), f4(d['nu']), f4(d['nuM_R95']),
                         ' / '.join(pc(d['kno_%s_R95' % r]) for r in ('combined', 'base', 'control', 'arm'))]))
assert all(d['m'] == '10' for d in ro)
L += ['', 'm = 10 in all rounds', '']


def v(op, con, fam):
    x = [b for b in bt if b['op'] == op and b['contrast'] == con and b['measure'] == 'det' and b['family'] == fam]
    assert len(x) == 1, (op, con, fam, len(x))
    return float(x[0]['point_pp']), float(x[0]['lo_pp']), float(x[0]['hi_pp'])


fams = [b['family'] for b in bt if b['op'] == 'OP-R95' and b['contrast'] == 'full-control' and b['measure'] == 'det']
L += ['Table 11b, OP-F2 and OP-SPEC columns (points)', 'family | control F2 | control SPEC | arm F2 | arm SPEC']
hw = 0
for f in fams:
    row = []
    for con in ('full-control', 'full-arm'):
        for op in ('OP-F2', 'OP-SPEC'):
            p, lo, hi = v(op, con, f)
            row.append(p)
            if not f.startswith('mean'):
                hw = max(hw, hi - p, p - lo)
    L.append('%s | %s' % (f, ' | '.join('%.2f' % x for x in row)))
L += ['', 'largest half-width of a per-family interval at OP-F2 / OP-SPEC: %.2f points' % hw]
fam13 = [f for f in fams if not f.startswith('mean') and 'joint' not in f and f not in config.TWINS]
sp = {f: v('OP-SPEC', 'full-control', f)[0] for f in fam13}
rest = [x for k, x in sp.items() if k != 'Exfiltration']
L += ['', 'OP-SPEC full-control, 13 families: n=%d, sum %.2f, Exfiltration %.2f, others min %.2f max %.2f mean %.3f' % (
    len(sp), sum(sp.values()), sp['Exfiltration'], min(rest), max(rest), sum(rest) / 12)]
open(os.path.join(wd, 'tables_11b_11d.txt'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
