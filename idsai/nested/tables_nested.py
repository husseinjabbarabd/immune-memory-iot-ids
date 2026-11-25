import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

R = config.out('nested')
b = json.load(open(os.path.join(R, 'nested_fpr.json')))
w = json.load(open(os.path.join(R, 'nested_wn.json')))
oc = json.load(open(os.path.join(R, 'curves_oneclass.json')))
F = [k for k in b if k not in ('meta', 'SYN group')]
SYN = [f for f in F if 'SYN' in f]
T = ['0.005', '0.01', '0.02', '0.05']
MS = ['immune', 'xgb', 'IF', 'OCSVM']


def rec(src, k, m, f):
    if m in ('IF', 'OCSVM'):
        o = oc[m][f]
        adr = sum(o['ADR'][s] for s in SYN) / 3 if k == 'SYN group' else o['ADR'][k]
        return dict(ADR=adr, EUR=adr, cal_fpr=o['cal_fpr'], FPR=o['FPR'])
    v = src[k][m][f]
    return v if v['attainable'] else None


def means(src, f, rounds):
    out = {}
    for m in MS:
        v = [rec(src, k, m, f) for k in rounds]
        v = [x for x in v if x]
        out[m] = dict(n=len(v), **({q: sum(x[q] for x in v) / len(v) for q in ('ADR', 'EUR', 'cal_fpr', 'FPR')} if v else {}))
    sh = [k for k in rounds if all(rec(src, k, m, f) for m in MS)]
    out['shared_rounds'] = sh
    out['shared'] = {m: {q: sum(rec(src, k, m, f)[q] for k in sh) / len(sh) for q in ('ADR', 'EUR', 'cal_fpr', 'FPR')}
                     for m in MS} if sh else {}
    return out


def g(x, d=2):
    s = '%.*f' % (d, x)
    return '100' if s == '100.00' else s


res = dict(nested={f: means(b, f, F) for f in T}, knob={f: means(w, f, F) for f in T})
L = []
for tag, src in (('nested', b), ('knob', w)):
    knob = tag == 'knob'
    for f in T:
        L.append('\n### %s, target %s%%  (rounds attainable of %d)\n' % (tag, g(100 * float(f), 1), len(F)))
        L.append('| Held-out attack | %sImm ADR | Imm EUR | Imm Cal | Imm Test | XGB ADR | XGB EUR | XGB Cal | XGB Test | '
                 'IF ADR | OC-SVM ADR |' % ('Imm lambda | ' if knob else ''))
        for k in F + ['SYN group']:
            row = [k]
            for m in ('immune', 'xgb'):
                v = rec(src, k, m, f)
                if knob and m == 'immune':
                    row.append(g(v['m']) if v else '–')
                if v:
                    row += [g(v['ADR']), g(v['EUR']), g(v['cal_fpr']), g(v['FPR'])]
                else:
                    pc = src[k][m][f]['path_cal_fpr_at_max'] if knob else src[k][m].get('path_cal_fpr')
                    if not knob and m == 'xgb':
                        pc = 100 * src[k][m]['path_cal_rows'] / 80000
                    row += ['n.a. (path %s)' % g(pc), '', '', '']
            row += [g(rec(src, k, 'IF', f)['ADR']), g(rec(src, k, 'OCSVM', f)['ADR'])]
            L.append('| ' + ' | '.join(row) + ' |')
        mm = res[tag][f]
        L.append('| Mean (attainable) | ' + ('| ' if knob else '') + ' | '.join(sum(
            (([g(mm[m]['ADR']), g(mm[m]['EUR']), g(mm[m]['cal_fpr']), g(mm[m]['FPR'])] if mm[m]['n'] else ['–'] * 4)
             for m in ('immune', 'xgb')), []))
            + ' | %s | %s |  n = %d / %d / %d / %d' % (g(mm['IF']['ADR']), g(mm['OCSVM']['ADR']), *(mm[m]['n'] for m in MS)))
        L.append('shared rounds: %s; %s' % (mm['shared_rounds'],
                                            {m: {q: round(v, 3) for q, v in d.items()} for m, d in mm['shared'].items()}))
        L.append('IF cal/test %.4f/%.4f, OCSVM cal/test %.4f/%.4f' % (oc['IF'][f]['cal_fpr'], oc['IF'][f]['FPR'],
                                                                      oc['OCSVM'][f]['cal_fpr'], oc['OCSVM'][f]['FPR']))
L.append('\n### nested path calibration FPR (immune, XGB rows/80000) and XGB best setting')
for k in F + ['SYN group']:
    L.append('%s: immune %.4f, xgb %.4f (%d rows), best %s, immune cv_acc %.4f' % (
        k, b[k]['immune']['path_cal_fpr'], 100 * b[k]['xgb']['path_cal_rows'] / 80000, b[k]['xgb']['path_cal_rows'],
        b[k]['xgb']['best'], b[k]['immune']['cv_acc']))
L.append('\n### knob seen-class accuracy (base lambda=1 -> chosen lambda) per target')
for f in T:
    for m in ('immune', 'xgb'):
        s = []
        for k in F:
            v = w[k][m][f]
            if v['attainable'] and v['m'] != 1.0:
                s.append('%s %.2f->%.2f (lambda %.2f)' % (k[:12], w[k][m]['base_seen_acc'], v['seen_acc'], v['m']))
        L.append('%s %s: %s' % (f, m, '; '.join(s) or 'all lambda = 1'))
open(os.path.join(R, 'tables_nested.txt'), 'w', encoding='utf-8').write('\n'.join(L))
json.dump(res, open(os.path.join(R, 'tables_nested.json'), 'w'), indent=1)
