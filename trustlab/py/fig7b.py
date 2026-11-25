import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config

pts = {}
for r in csv.DictReader(open(os.path.join(config.RES, 'common_far', 'curves.csv'))):
    if r['in_shared_range'] == '1':
        pts.setdefault((r['version'], r['detector']), []).append((float(r['mean_test_far']), float(r['mean_detection'])))
ver = {'port-using': 'with ports', 'port-free': 'without ports'}
det = {'full rule': 'combined rule', 'control': 'threshold-matched control', 'arm alone': 'arm alone'}
sty = {'full rule': '-', 'control': '--', 'arm alone': ':'}
col = {'port-using': 'tab:blue', 'port-free': 'tab:red'}
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for v in ('port-using', 'port-free'):
    for k in ('full rule', 'control', 'arm alone'):
        P = sorted(pts[(v, k)])
        ax.plot([100 * x for x, _ in P], [100 * y for _, y in P], sty[k], color=col[v], label='%s, %s' % (ver[v], det[k]))
ax.axvline(1.169, color='grey', lw=0.8)
ax.set_xlabel('mean test false-alarm rate (%)')
ax.set_ylabel('mean held-out detection (%)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(config.out('figures'), 'fig07b.png'), dpi=300)
