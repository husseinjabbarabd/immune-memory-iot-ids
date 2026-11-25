import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import offset_copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

GREEN, INK = '#059669', '#1F1F1F'
ABBR = ['ARP', 'SSH', 'MAC', 'ICMP', 'Frag', 'Nor', 'SYN', 'SA', 'SAR', 'Null', 'UDP']


def confusion(ax, counts, labels):
    pct = counts / counts.sum(axis=1, keepdims=True) * 100
    ax.imshow(pct, cmap='Blues', vmin=0, vmax=100)
    n = len(labels)
    for i in range(n):
        for j in range(n):
            v = pct[i, j]
            if counts[i, j] == 0:
                continue
            if i != j and v < 1.0:
                ax.plot(j, i, 'o', ms=2.2, color=INK)
                continue
            txt = '100' if v >= 99.95 else ('%.0f' % v if i != j else '%.1f' % v)
            ax.text(j, i, txt, ha='center', va='center', color='white' if v >= 60 else INK)
    ax.set_xticks(range(n), labels, rotation=90, ha='right', va='center', rotation_mode='anchor')
    ax.set_yticks(range(n), labels)
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return pct


def panel(ax, letter, x):
    tr = offset_copy(ax.transAxes, fig=ax.figure, x=0, y=7, units='points')
    ax.text(x, 1.0, '(%s)' % letter, transform=tr, fontweight='bold', ha='right', va='bottom')


s = np.load(os.path.join(config.WORK, 'split.npz'), allow_pickle=True)
classes = [str(c) for c in s['classes']]
cm = np.load(os.path.join(config.WORK, 'final_test_eval.npz'))['cm']
maj = json.load(open(os.path.join(config.RES, 'base', 'quick_checks.json')))['ceiling_test_per_class_recall']
maj = np.array([maj[c] for c in classes])
rec = np.round(np.diag(cm) / cm.sum(axis=1), 4)
assert cm.sum() == 200000 and round(np.trace(cm) / cm.sum() * 100, 3) == 92.767
assert rec[0] == 0.4672 and maj[0] == 0.6282

plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.linewidth': 0.6, 'grid.color': '#D9D9D9',
                     'grid.linewidth': 0.5, 'legend.frameon': False, 'pdf.fonttype': 42})
fig = plt.figure(figsize=(17.0 / 2.54, 3.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1], wspace=0.2)
ax = fig.add_subplot(gs[0])
pct = confusion(ax, cm, ABBR)
panel(ax, 'a', -0.04)
ax = fig.add_subplot(gs[1])
y = np.arange(len(ABBR))[::-1]
ax.barh(y + 0.19, rec, height=0.36, color=GREEN, label='Immune memory, test recall')
ax.barh(y - 0.19, maj, height=0.36, color='white', edgecolor=INK, lw=0.7, hatch='////',
        label='Majority-label recall\n(same test rows)')
ax.set_yticks(y, ABBR)
ax.set_ylim(-0.6, len(y) - 0.4)
ax.set_xlim(0.4, 1.55)
ax.set_xticks([0.4, 0.6, 0.8, 1.0])
ax.spines['bottom'].set_bounds(0.4, 1.0)
ax.set_xlabel('Recall')
ax.grid(axis='x')
ax.set_axisbelow(True)
for yy, r, c in zip(y, rec, maj):
    if r < 0.99:
        ax.text(1.02, yy, '%.4f / %.4f' % (r, c), va='center')
ax.legend(loc='upper left', bbox_to_anchor=(-0.22, -0.14), ncol=1, handlelength=1.6)
panel(ax, 'b', -0.16)

od = config.out('figures')
pd.DataFrame(cm, index=classes, columns=['pred_' + c for c in classes]).to_csv(os.path.join(od, 'fig04a_counts.csv'))
pd.DataFrame(pct, index=classes, columns=['pred_' + c for c in classes]).to_csv(os.path.join(od, 'fig04a_rowpct.csv'),
                                                                                 float_format='%.10g')
pd.DataFrame({'class': classes, 'test_recall': rec, 'majority_label_recall': maj}).to_csv(
    os.path.join(od, 'fig04b_values.csv'), index=False, float_format='%.10g')
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(od, 'fig04.' + ext), bbox_inches='tight', pad_inches=0.02, dpi=600)
