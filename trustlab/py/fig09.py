import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

import config

INK = '#1F1F1F'
ABBR = ['API', 'Ben', 'BF', 'BO', 'C2', 'DDoS', 'DNS', 'DoS', 'Ev', 'Exf', 'Expl', 'MITM', 'PS', 'SL', 'TLS', 'Web']
C = config.CLASSES
T = loadmat(os.path.join(config.RES, 'pipeline', 'test_rows.mat'), variable_names=['sT', 'pAll', 'yTe', 'taus'])
s, y, taus = T['sT'].ravel(), T['yTe'].ravel().astype(int), T['taus'].ravel()
p = T['pAll'].ravel().astype(int)
p[s < taus[2]] = C.index('Benign') + 1
cm = np.zeros((16, 16), np.int64)
np.add.at(cm, (y - 1, p - 1), 1)
assert cm[C.index('DoS'), C.index('Slowloris')] == 8285 and cm[C.index('Exploitation'), C.index('Benign')] == 7814

plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10, 'pdf.fonttype': 42})
W, H = 16.0 / 2.54, 9.0 / 2.54
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0.62 / W, 0.44 / H, (W - 0.66) / W, (H - 0.46) / H])
pct = cm / cm.sum(axis=1, keepdims=True) * 100
ax.imshow(pct, cmap='Blues', vmin=0, vmax=100)
for i in range(16):
    for j in range(16):
        v = pct[i, j]
        if cm[i, j] == 0:
            continue
        if i != j and v < 5.0:
            ax.plot(j, i, 'o', ms=2.2, color=INK)
            continue
        txt = '100' if v >= 99.95 else ('%.0f' % v if i != j else '%.1f' % v)
        ax.text(j, i, txt, ha='center', va='center', color='white' if v >= 60 else INK)
ax.set_xticks(range(16), ABBR)
ax.set_yticks(range(16), ABBR)
ax.set_xlabel('Predicted class')
ax.set_ylabel('True class')
for sp in ax.spines.values():
    sp.set_visible(False)
ax.tick_params(length=0)
ax.set_aspect('auto')
od = config.out('figures')
pd.DataFrame(cm, index=C, columns=['pred_' + c for c in C]).to_csv(os.path.join(od, 'fig09_counts.csv'))
pd.DataFrame(pct, index=C, columns=['pred_' + c for c in C]).to_csv(os.path.join(od, 'fig09_rowpct.csv'), float_format='%.10g')
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(od, 'fig09.' + ext), bbox_inches='tight', pad_inches=0.02, dpi=600)
