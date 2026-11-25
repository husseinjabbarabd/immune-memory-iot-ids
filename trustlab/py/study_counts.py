import csv
import json
import os
import sys

import config

D = config.out('study')
rows = list(csv.DictReader(open(os.path.join(D, 'perclass_counts.csv'))))
J = json.load(open(os.path.join(D, 'reweight_all.json')))
old = json.load(open(os.path.join(D, 'op_metrics.json')))
nB, nA = 514478, 385856
WB, WA = 436332 / nB, 597862 / nA


def f(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


def g(r, k):
    return int(r[k])


def calc(R, wb, wa):
    att = [r for r in R if r['class'] != 'Benign']
    b = [r for r in R if r['class'] == 'Benign'][0]
    kept_att = sum(g(r, 'n') - g(r, 'fwd') for r in att)
    F16 = []
    for r in att:
        col = wa * g(r, 'c') + wb * g(r, 'FPb') + wa * g(r, 'FPa')
        F16.append(f(wa * g(r, 'c') / col, g(r, 'c') / g(r, 'fwd')))
    colB = wb * g(b, 'c') + wa * g(b, 'FPa')
    F16.append(f(wb * g(b, 'c') / colB, g(b, 'c') / g(b, 'fwd')))
    fwd_tot = wb * g(b, 'fwd') + wa * sum(g(r, 'fwd') for r in att)
    fwd_acc = (wb * g(b, 'c') + wa * sum(g(r, 'c') for r in att)) / fwd_tot
    Fa = [f(g(r, 'c') / (g(r, 'c') + g(r, 'FPa')), g(r, 'c') / g(r, 'fwd')) for r in att]
    Fb = [f(wa * g(r, 'c') / (wa * g(r, 'c') + wb * g(r, 'FPb') + wa * g(r, 'FPa')), g(r, 'c') / g(r, 'fwd')) for r in att]
    att_acc = sum(g(r, 'c') for r in att) / sum(g(r, 'fwd') for r in att)
    E = []
    for r in att:
        col = wa * g(r, 'c') + wb * g(r, 'FPb') + wa * g(r, 'FPa')
        E.append(f(wa * g(r, 'c') / col, g(r, 'c') / g(r, 'n')))
    tpB = wb * (g(b, 'n') - g(b, 'fwd') + g(b, 'c'))
    E.append(f(tpB / (tpB + wa * (g(b, 'FPa') + kept_att)), tpB / (wb * g(b, 'n'))))
    tot = wb * nB + wa * nA
    e2e_acc = (tpB + wa * sum(g(r, 'c') for r in att)) / tot
    fa = wb * sum(g(r, 'FPb') for r in att)
    return dict(fwd16_acc=fwd_acc, fwd16_mf1=sum(F16) / 16, att15_acc=att_acc, att15_mf1_attack_rows=sum(Fa) / 15,
                att15_mf1_with_benign_fp=sum(Fb) / 15, e2e_acc=e2e_acc, e2e_mf1=sum(E) / 16, false_alarm_share=fa / tot,
                fa_count=sum(g(r, 'FPb') for r in att), fa_check=g(b, 'fwd') - g(b, 'c'))


bad = 0
L = []
for op in ('OP-SPEC', 'OP-F2', 'OP-R95'):
    R = [r for r in rows if r['op'] == op]
    u, w = calc(R, 1.0, 1.0), calc(R, WB, WA)
    ref = {**J[op]['reweighted'], **J[op]['reweighted_reproduced']}
    L.append(op)
    for k in ('fwd16_acc', 'fwd16_mf1', 'att15_acc', 'att15_mf1_attack_rows', 'att15_mf1_with_benign_fp',
              'e2e_acc', 'e2e_mf1', 'false_alarm_share'):
        ok = abs(u[k] - old[op][k]) < 1e-12
        bad += not ok
        line = '  %-26s own %.6f %s' % (k, u[k], 'ok' if ok else 'DIFF %.9f' % old[op][k])
        if k in ref:
            okw = abs(w[k] - ref[k]) < 1e-12
            bad += not okw
            line += ' | ref-mix %.6f %s' % (w[k], 'ok' if okw else 'DIFF %.9f' % ref[k])
        L.append(line)
    L.append('  false alarms: sum FPb %d, Benign fwd - c %d' % (u['fa_count'], u['fa_check']))
    bad += u['fa_count'] != u['fa_check']
L.append('ALL MATCH' if bad == 0 else 'MISMATCHES %d' % bad)
open(os.path.join(D, 'counts_check.txt'), 'w').write('\n'.join(L) + '\n')
if bad:
    sys.exit('counts do not reproduce the table values')
