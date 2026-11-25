import json
import os
import sys
from decimal import Decimal, ROUND_HALF_UP

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

R = config.out('nested')
b = json.load(open(os.path.join(R, 'nested_fpr.json')))
w = json.load(open(os.path.join(R, 'nested_wn.json')))
oc = json.load(open(os.path.join(R, 'curves_oneclass.json')))
F = [k for k in b if k not in ('meta', 'SYN group')]
SYN = [f for f in F if 'SYN' in f]
T = ['0.005', '0.01', '0.02', '0.05']
TL = {'0.005': '0.5%', '0.01': '1%', '0.02': '2%', '0.05': '5%'}
D = lambda x: Decimal('%.6f' % x)


def g(x, n=2):
    if isinstance(x, str):
        return x
    d = D(x).normalize()
    if n == 2 and -d.as_tuple().exponent == 3 and d.as_tuple().digits[-1] == 5:
        return str(d)
    s = str(D(x).quantize(Decimal(1).scaleb(-n), ROUND_HALF_UP))
    return '100' if s in ('100.00', '100.000') else s


def rec(src, k, m, f):
    if m in ('IF', 'OCSVM'):
        o = oc[m][f]
        adr = sum(D(o['ADR'][s]) for s in SYN) / 3 if k == 'SYN group' else D(o['ADR'][k])
        return dict(ADR=adr, EUR=adr, cal_fpr=D(o['cal_fpr']), FPR=D(o['FPR']))
    v = src[k][m][f]
    return {q: (D(v[q]) if isinstance(v[q], float) or isinstance(v[q], int) else v[q])
            for q in v if q != 'per_class'} if v['attainable'] else None


def mean(src, m, f, ks):
    v = [rec(src, k, m, f) for k in ks]
    v = [x for x in v if x]
    return len(v), ({q: sum(x[q] for x in v) / len(v) for q in ('ADR', 'EUR', 'cal_fpr', 'FPR')} if v else None)


def line(cells):
    return '| ' + ' | '.join(cells) + ' |'


out = []
for tag, src in (('(a) Nested comparison (primary)', b),
                 ('(b) Additional operating protocol: Normal-weight knob lambda on the classification path', w)):
    out.append('\n' + tag + '\n')
    out.append(line(['Target and rounds', 'Rounds averaged: immune / IF / OC-SVM / XGB', 'Immune ADR', 'Immune EUR',
                     'Immune Cal.', 'Immune Test', 'IF ADR', 'IF Test', 'OC-SVM ADR', 'OC-SVM Test', 'XGB ADR', 'XGB EUR',
                     'XGB Cal.', 'XGB Test']))
    out.append('|' + '---|' * 14)
    for f in T:
        ns = {m: mean(src, m, f, F) for m in ('immune', 'IF', 'OCSVM', 'xgb')}
        sh = [k for k in F if all(rec(src, k, m, f) for m in ('immune', 'IF', 'OCSVM', 'xgb'))]
        rows = [('attainable', ns)]
        if len(sh) < len(F):
            rows.append(('shared', {m: mean(src, m, f, sh) for m in ns}))
        for lab, d in rows:
            if lab == 'shared' and not sh:
                out.append(line(['%s, shared' % TL[f], '0 / 0 / 0 / 0'] + ['–'] * 12))
                continue
            lab2 = TL[f] if len(sh) == len(F) else '%s, %s' % (TL[f], lab)
            c = [lab2, ' / '.join(str(d[m][0]) for m in ('immune', 'IF', 'OCSVM', 'xgb'))]
            for m, qs in (('immune', 'ADR EUR cal_fpr FPR'), ('IF', 'ADR FPR'), ('OCSVM', 'ADR FPR'), ('xgb', 'ADR EUR cal_fpr FPR')):
                c += [g(d[m][1][q]) if d[m][1] else '–' for q in qs.split()]
            out.append(line(c))
        if len(sh) < len(F):
            out.append('<!-- %s shared rounds: %s -->' % (TL[f], sh))
for i, f in enumerate(T):
    out.append('\n(%s) Nested comparison, %s target\n' % ('abcd'[i], TL[f]))
    out.append(line(['Held-out attack', 'Immune ADR', 'Immune EUR', 'Immune Cal.', 'Immune Test', 'XGB ADR', 'XGB EUR',
                     'XGB Cal.', 'XGB Test', 'IF ADR', 'OC-SVM ADR']))
    out.append('|' + '---|' * 11)
    for k in F + ['Mean', 'SYN group']:
        c = [k if k != 'SYN group' else 'SYN group ᵈ']
        if k == 'Mean':
            c = ['**Mean** (rounds: %d / %d)' % (mean(b, 'immune', f, F)[0], mean(b, 'xgb', f, F)[0])]
            for m in ('immune', 'xgb'):
                n, d = mean(b, m, f, F)
                c += ['**%s**' % g(d[q]) for q in ('ADR', 'EUR', 'cal_fpr', 'FPR')] if d else ['–'] * 4
            c += ['**%s**' % g(mean(b, m, f, F)[1]['ADR']) for m in ('IF', 'OCSVM')]
        else:
            for m in ('immune', 'xgb'):
                v = rec(b, k, m, f)
                if v:
                    c += [g(v[q]) for q in ('ADR', 'EUR', 'cal_fpr', 'FPR')]
                else:
                    p = b[k][m]['path_cal_fpr'] if m == 'immune' else 100 * b[k][m]['path_cal_rows'] / 80000
                    s = g(p)
                    if float(s) <= 100 * float(f):
                        s = g(p, 3)
                    c += ['n.a.', '–', '(%s)' % s, '–']
            c += [g(rec(b, k, m, f)['ADR']) for m in ('IF', 'OCSVM')]
        out.append(line(c))
f = '0.02'
out.append('\n(e) Additional operating protocol (knob lambda), 2% target, immune detector; XGBoost keeps lambda = 1 in every '
           'round, so its values are those of panel (c)\n')
out.append(line(['Held-out attack', 'lambda', 'Immune ADR', 'Immune EUR', 'Immune Cal.', 'Immune Test',
                 'Seen-class accuracy, lambda = 1', 'Seen-class accuracy, chosen lambda']))
out.append('|' + '---|' * 8)
for k in F + ['Mean', 'SYN group']:
    if k == 'Mean':
        n, d = mean(w, 'immune', f, F)
        sa = sum(D(w[x]['immune']['base_seen_acc']) for x in F) / 10
        sb = sum(D(w[x]['immune'][f]['seen_acc']) for x in F) / 10
        c = (['**Mean** (rounds: %d)' % n, '–'] + ['**%s**' % g(d[q]) for q in ('ADR', 'EUR', 'cal_fpr', 'FPR')]
             + ['**%s**' % g(sa), '**%s**' % g(sb)])
    else:
        v = w[k]['immune'][f]
        c = [k if k != 'SYN group' else 'SYN group ᵈ', g(v['m']), g(v['ADR']), g(v['EUR']), g(v['cal_fpr']), g(v['FPR']),
             g(w[k]['immune']['base_seen_acc']), g(v['seen_acc'])]
    out.append(line(c))
v = json.load(open(os.path.join(R, 'nested_xgb_def.json')))
out.append('\nTable 9(a) extra rows, XGBoost variant ᵉ\n')
for f in T:
    n, d = mean(v, 'xgb', f, F)
    c = (['%s, XGB variant ᵉ' % TL[f], '– / – / – / %d' % n] + ['–'] * 8
         + ([g(d[q]) for q in ('ADR', 'EUR', 'cal_fpr', 'FPR')] if d else ['–'] * 4))
    out.append(line(c))
for i, f in enumerate(T):
    out.append('\n(%s) XGBoost variant (four settings at library defaults), nested, %s target\n' % ('hijk'[i], TL[f]))
    out.append(line(['Held-out attack', 'XGB ADR', 'XGB EUR', 'XGB Cal.', 'XGB Test']))
    out.append('|' + '---|' * 5)
    for k in F + ['Mean', 'SYN group']:
        if k == 'Mean':
            n, d = mean(v, 'xgb', f, F)
            c = ['**Mean** (rounds: %d)' % n] + (['**%s**' % g(d[q]) for q in ('ADR', 'EUR', 'cal_fpr', 'FPR')] if d else ['–'] * 4)
        else:
            x = rec(v, k, 'xgb', f)
            c = [k if k != 'SYN group' else 'SYN group ᵈ']
            if x:
                c += [g(x[q]) for q in ('ADR', 'EUR', 'cal_fpr', 'FPR')]
            else:
                p = 100 * v[k]['xgb']['path_cal_rows'] / 80000
                s = g(p)
                if float(s) <= 100 * float(f):
                    s = g(p, 3)
                c += ['n.a.', '–', '(%s)' % s, '–']
        out.append(line(c))
out.append('<!-- variant best grid point per round: %s; nested: %s -->' % ({k: v[k]['xgb']['best'] for k in v if k != 'meta'},
                                                                         {k: b[k]['xgb']['best'] for k in b if k != 'meta'}))
out.append('<!-- variant path cal rows: %s -->' % {k: v[k]['xgb']['path_cal_rows'] for k in v if k != 'meta'})
out.append('\n<!-- facts')
for f in T:
    for m in ('immune', 'xgb'):
        nn = [k for k in F if w[k][m][f]['attainable']]
        ms = sorted({w[k][m][f]['m'] for k in nn})
        out.append('knob %s %s attainable %d, lambda values %s' % (f, m, len(nn), ms))
out.append('one-class cal FPR: ' + str({n: {TL[f]: g(oc[n][f]['cal_fpr'], 4) for f in T} for n in oc}))
out.append('one-class test FPR: ' + str({n: {TL[f]: g(oc[n][f]['FPR'], 3) for f in T} for n in oc}))
for f in ('0.02', '0.05'):
    for m in ('immune', 'xgb'):
        if b['SYN group'][m][f]['attainable']:
            out.append('SYN group %s %s per class [EUR, ADR]: %s FPR %s' % (f, m, b['SYN group'][m][f]['per_class'],
                                                                            b['SYN group'][m][f]['FPR']))
out.append('-->')
open(os.path.join(R, 'tables_paper.txt'), 'w', encoding='utf-8').write('\n'.join(out))
