import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

GREEN, BLUE = '#059669', '#2563EB'
R = os.path.join(config.RES, 'base')
a = 100 * np.array(json.load(open(os.path.join(R, 'cv10.json')))['folds'])
x = 100 * np.array(json.load(open(os.path.join(R, 'xgbT_cv10.json')))['folds'])
assert len(a) == len(x) == 10 and round(a.mean(), 4) == 92.8185

plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.linewidth': 0.6, 'grid.color': '#D9D9D9',
                     'grid.linewidth': 0.5, 'legend.frameon': False, 'pdf.fonttype': 42})
fig, ax = plt.subplots(figsize=(8.1 / 2.54, 2.85))
f = np.arange(1, 11)
ax.plot(f, a, '-o', color=GREEN, ms=4, lw=1.2, label='Immune memory')
ax.plot(f, x, '-s', color=BLUE, ms=4, lw=1.2, label='XGBoost, tuned rerun')
ax.set_xticks(f)
ax.set_xlabel('Fold')
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(92.2, 93.0)
ax.grid(axis='y')
ax.legend(loc='lower left', ncol=1, handlelength=1.8)
od = config.out('figures')
pd.DataFrame({'fold': f, 'immune': a, 'xgboost_tuned': x, 'difference': a - x}).to_csv(
    os.path.join(od, 'fig05_values.csv'), index=False, float_format='%.10g')
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(od, 'fig05.' + ext), bbox_inches='tight', pad_inches=0.02, dpi=600)
