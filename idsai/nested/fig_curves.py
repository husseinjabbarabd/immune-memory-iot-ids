import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

R = config.out('nested')
w = json.load(open(os.path.join(R, 'nested_wn.json')))
oc = json.load(open(os.path.join(R, 'curves_oneclass.json')))
F = [k for k in w if k not in ('meta', 'SYN group')]
short = dict(zip(F, ['ARP', 'SSH', 'MAC', 'ICMP', 'Frag', 'SYN', 'SA', 'SAR', 'Null', 'UDP']))


def curves(k):
    c = {}
    for m, n in (('immune', 'Immune'), ('xgb', 'XGBoost')):
        a = np.array(w[k][m]['curve_m1'])
        c[n] = (a[:, 1], a[:, 2], a[:, 3])
    for m, n in (('IF', 'Isolation Forest'), ('OCSVM', 'One-Class SVM')):
        cols = oc[m]['curve']['columns']
        a = np.array(oc[m]['curve']['points'])
        c[n] = (a[:, 1], a[:, cols.index(k)], a[:, cols.index(k)])
    for n in c:
        o = np.argsort(c[n][0], kind='stable')
        c[n] = tuple(x[o] for x in c[n])
    return c


sty = {'Immune': ('#059669', '-'), 'XGBoost': ('#2563eb', '-'), 'Isolation Forest': ('#d97706', ':'),
       'One-Class SVM': ('#404040', '-.')}
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'], 'font.size': 9})
fig, ax = plt.subplots(4, 3, figsize=(7.2, 9.0), constrained_layout=True)
out = {'ranges': {}, 'mean': {}}
cs = {k: curves(k) for k in F}
for i, k in enumerate(F):
    c = cs[k]
    lo = max(v[0].min() for v in c.values())
    hi = min(v[0].max() for v in c.values())
    out['ranges'][k] = [float(lo), float(hi)]
    a = ax.flat[i]
    for n, (x, adr, eur) in c.items():
        s = (x >= lo) & (x <= hi)
        col, ls = sty[n]
        a.plot(x[s], adr[s], color=col, ls=ls, lw=1.4, label=n + ' ADR')
        if n in ('Immune', 'XGBoost'):
            a.plot(x[s], eur[s], color=col, ls='--', lw=0.9, alpha=0.8, label=n + ' EUR')
    a.set_title('%s (%.2f–%.2f%%)' % (short[k], lo, hi), fontsize=9)
    a.set_ylim(-3, 103)
    a.grid(alpha=0.3)
lo = max(r[0] for r in out['ranges'].values())
hi = min(r[1] for r in out['ranges'].values())
g = np.round(np.arange(np.ceil(lo * 20) / 20, hi + 1e-9, 0.05), 2)
a = ax.flat[10]
for n in sty:
    adr = np.mean([np.interp(g, *cs[k][n][:2]) for k in F], 0)
    eur = np.mean([np.interp(g, cs[k][n][0], cs[k][n][2]) for k in F], 0)
    out['mean'][n] = dict(fpr=g.tolist(), ADR=adr.tolist(), EUR=eur.tolist())
    col, ls = sty[n]
    a.plot(g, adr, color=col, ls=ls, lw=1.6)
    if n in ('Immune', 'XGBoost'):
        a.plot(g, eur, color=col, ls='--', lw=1.0, alpha=0.8)
a.set_title('Mean of 10 (%.2f–%.2f%%)' % (g[0], g[-1]), fontsize=9)
a.set_ylim(-3, 103)
a.grid(alpha=0.3)
out['common'] = [float(g[0]), float(g[-1])]
h, l = ax.flat[0].get_legend_handles_labels()
ax.flat[11].axis('off')
ax.flat[11].legend(h, l, loc='center', frameon=False, fontsize=8)
for a in ax[:, 0]:
    a.set_ylabel('Detected (%)')
for a in ax.flat[:11]:
    a.set_xlabel('Test FPR (%)', fontsize=8)
    a.xaxis.set_major_locator(MaxNLocator(integer=True))
for e in ('png', 'pdf', 'svg'):
    fig.savefig(os.path.join(R, 'fig_curves.' + e), dpi=300)
json.dump(out, open(os.path.join(R, 'curves_shared.json'), 'w'), indent=1)
