import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

R = config.out('nested')
v = json.load(open(os.path.join(R, 'nested_xgb_def.json')))
b = json.load(open(os.path.join(R, 'nested_fpr.json')))
T = ['0.005', '0.01', '0.02', '0.05']
single = [k for k in b if k not in ('meta', 'SYN group')]
out = dict(rounds=single)


def g(x, t, q):
    return x[t].get(q) if x[t]['attainable'] else None


def mean(xs):
    xs = [a for a in xs if a is not None]
    return (sum(xs) / len(xs), len(xs)) if xs else (None, 0)


rows = {}
for name, src in (('nested', b), ('def', v)):
    r = {}
    for t in T:
        r[t] = dict(n=sum(src[h]['xgb'][t]['attainable'] for h in single),
                    ADR=mean([g(src[h]['xgb'], t, 'ADR') for h in single])[0],
                    EUR=mean([g(src[h]['xgb'], t, 'EUR') for h in single])[0],
                    cal=mean([g(src[h]['xgb'], t, 'cal_fpr') for h in single])[0],
                    FPR=mean([g(src[h]['xgb'], t, 'FPR') for h in single])[0],
                    per=[(h, g(src[h]['xgb'], t, 'ADR'), g(src[h]['xgb'], t, 'EUR'), g(src[h]['xgb'], t, 'cal_fpr'),
                          g(src[h]['xgb'], t, 'FPR')) for h in single],
                    syn=(g(src['SYN group']['xgb'], t, 'ADR'), g(src['SYN group']['xgb'], t, 'EUR'),
                         g(src['SYN group']['xgb'], t, 'cal_fpr'), g(src['SYN group']['xgb'], t, 'FPR')))
    r['best'] = {h: src[h]['xgb']['best'] for h in src if h != 'meta'}
    r['path_cal'] = {h: src[h]['xgb']['path_cal_fpr'] for h in src if h != 'meta'}
    rows[name] = r
out['xgb'] = rows
im = {h: b[h]['immune'] for h in b if h != 'meta'}
i5 = {h: (im[h]['0.05']['ADR'], im[h]['0.05']['EUR']) for h in single}
iA = mean([a for a, e in i5.values()])[0]
iE = mean([e for a, e in i5.values()])[0]
imm2 = sum(im[h]['0.02']['attainable'] for h in single)
C = {}
for name in ('nested', 'def'):
    x = rows[name]
    p5 = {h: (a, e) for h, a, e, c, f in x['0.05']['per']}
    C[name] = dict(
        C1=iA > x['0.05']['ADR'], C2=all(i5[h][0] >= p5[h][0] for h in single),
        C2_fail=[h for h in single if i5[h][0] < p5[h][0]], C3=iE > x['0.05']['EUR'],
        C4=x['0.05']['EUR'] < 0.5 * x['0.05']['ADR'],
        C5=p5['Brute Force SSH'][0] < 5 and p5['UDP port scan'][0] < 5,
        C5_vals=(p5['Brute Force SSH'][0], p5['UDP port scan'][0]),
        C6=[x[t]['n'] for t in T[:3]], C6_2pct=x['0.02']['n'] >= 9 and imm2 == 1,
        C7=all(x[t]['syn'][0] == 100 and x[t]['syn'][1] < 1 for t in ('0.02', '0.05')),
        C7_vals=[x[t]['syn'] for t in ('0.02', '0.05')])
out['immune'] = dict(ADR5=iA, EUR5=iE, attainable2=imm2)
out['checks'] = C
json.dump(out, open(os.path.join(R, 'xgb_def_checks.json'), 'w'), indent=1)
