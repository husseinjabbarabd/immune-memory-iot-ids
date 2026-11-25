import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

R = config.out('fpr')
d = json.load(open(os.path.join(R, 'fpr_table.json')))


def v(x):
    return '100' if abs(x - 100) < 1e-9 else '%.2f' % x


L = ['TABLE 9']
for f, lab in (('0.005', '0.5%'), ('0.01', '1%'), ('0.02', '2%'), ('0.05', '5%')):
    for key, nm in (('means_own_attainable', 'attainable rounds'), ('means_shared', 'shared rounds')):
        M = d[key][f]
        n = ' / '.join(str(M[m]['n_rounds']) for m in ('immune', 'IF', 'OCSVM', 'xgb'))
        cells = [v(M['immune'][k]) for k in ('ADR', 'EUR', 'FPR')] + [v(M['IF']['ADR']), v(M['IF']['FPR']),
                 v(M['OCSVM']['ADR']), v(M['OCSVM']['FPR'])] + [v(M['xgb'][k]) for k in ('ADR', 'EUR', 'FPR')]
        L.append('| %s, %s | %s | %s |' % (lab, nm, n, ' | '.join(cells)))
L.append('TABLE 10')
for f in ('0.02', '0.05'):
    L.append(f)
    P = d['per_round'][f]
    for h, r in P.items():
        i, x = r['immune'], r['xgb']
        L.append('| %s | %s |' % (h, ' | '.join([v(i['ADR']), v(i['EUR']), v(i['cal']), v(i['FPR']), v(x['ADR']), v(x['EUR']),
                                                 v(x['cal']), v(x['FPR']), v(r['IF']['ADR']), v(r['OCSVM']['ADR'])])))
    M = d['means_own_attainable'][f]
    L.append('| **Mean** | %s |' % ' | '.join('**%s**' % v(c) for c in [M['immune']['ADR'], M['immune']['EUR'], M['immune']['cal'],
             M['immune']['FPR'], M['xgb']['ADR'], M['xgb']['EUR'], M['xgb']['cal'], M['xgb']['FPR'], M['IF']['ADR'], M['OCSVM']['ADR']]))
open(os.path.join(R, 'fpr_rows.txt'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
