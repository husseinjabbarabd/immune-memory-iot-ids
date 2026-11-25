import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score, roc_curve

import config

GREEN, ORANGE, INK = '#059669', '#F59E0B', '#1F1F1F'
T = loadmat(os.path.join(config.RES, 'pipeline', 'test_rows.mat'), variable_names=['sT', 'yTe', 'taus'])
s, y, taus = T['sT'].ravel(), T['yTe'].ravel().astype(int), T['taus'].ravel()
att = y != config.CLASSES.index('Benign') + 1
assert round(roc_auc_score(att, s), 6) == 0.998439
fpr, tpr, _ = roc_curve(att, s)
ops = pd.DataFrame({'opPoint': ['OP-SPEC', 'OP-F2', 'OP-R95'], 'tau': taus,
                    'FPR': [np.mean(s[~att] >= t) for t in taus], 'TPR': [np.mean(s[att] >= t) for t in taus]})
assert round(ops.TPR[2], 4) == 0.9509

plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.linewidth': 0.6, 'grid.color': '#D9D9D9',
                     'grid.linewidth': 0.5, 'legend.frameon': False, 'pdf.fonttype': 42})
fig, ax = plt.subplots(figsize=(8.1 / 2.54, 2.7))
ax.plot(np.clip(fpr, 1e-5, None), tpr, color=GREEN, lw=1.2, label='Ours, phase 1')
for m, (_, o) in zip('os^', ops.iterrows()):
    ax.plot(o.FPR, o.TPR, m, color=GREEN, mec=INK, mew=0.6, ms=6)
    ax.annotate(o.opPoint, (o.FPR, o.TPR), xytext=(4, -12), textcoords='offset points')
ref = (1 - 0.8229, 0.9498)
ax.plot(*ref, 'D', color=ORANGE, mec=INK, mew=0.6, ms=6, label='[25], t = 0.445')
ax.set_xscale('log')
ax.set_xlim(1e-4, 1)
ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1], ['0.0001', '0.001', '0.01', '0.1', '1'])
ax.minorticks_off()
ax.set_ylim(0.80, 1.005)
ax.set_xlabel('False-positive rate (benign flows), log scale')
ax.set_ylabel('True-positive rate')
ax.grid(True, which='major')
ax.legend(loc='lower right', ncol=1, handlelength=1.5)
od = config.out('figures')
pd.concat([ops, pd.DataFrame([{'opPoint': '[25] (1 - specificity, recall)', 'tau': 0.445, 'FPR': ref[0], 'TPR': ref[1]}])]).to_csv(
    os.path.join(od, 'fig08_values.csv'), index=False, float_format='%.10g')
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(od, 'fig08.' + ext), bbox_inches='tight', pad_inches=0.02, dpi=600)
