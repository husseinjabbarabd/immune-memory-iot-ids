import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon, ttest_rel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.chdir(config.WORK)
R = config.out('base')
a = np.array(json.load(open(os.path.join(R, 'cv10.json')))['folds'])
out = {'ais_mean': 100 * a.mean(), 'ais_std': 100 * a.std()}
for n in ['xgb_cv10', 'xgbT_cv10']:
    x = np.array(json.load(open(os.path.join(R, n + '.json')))['folds'])
    d = 100 * (a - x)
    out[n] = dict(mean=100 * x.mean(), std=100 * x.std(), wins=int((d > 0).sum()), diff_mean=d.mean(), diff_min=d.min(),
                  diff_max=d.max(), wilcoxon_p=float(wilcoxon(a, x).pvalue), ttest_p=float(ttest_rel(a, x).pvalue),
                  folds=x.tolist())


def wilson(k, n, z=1.959963984540054):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [100 * (c - h), 100 * (c + h)]


dup = json.load(open(os.path.join(R, 'dup.json')))['Unseen signature']
n = dup['n']
ka, kx = round(dup['ais_acc'] * n), round(dup['xgb_acc'] * n)
out['unseen'] = dict(n=n, ais=ka, xgb=kx, ais_wilson=wilson(ka, n), xgb_wilson=wilson(kx, n))
s = np.load('split.npz', allow_pickle=True)
ytr = s['ytr']
w = np.array(json.load(open(config.TUNED))['w'])
cnt = np.bincount(ytr, minlength=11)
out['prior'] = dict(cls=str(s['classes'][(w * cnt).argmax()]), normal_train=int(cnt[list(s['classes']).index('Normal')]),
                    train=int(len(ytr)))
json.dump(out, open(os.path.join(R, 'cv_paired.json'), 'w'), indent=1)
