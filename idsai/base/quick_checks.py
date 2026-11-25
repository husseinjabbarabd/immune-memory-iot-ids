import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from ais import layers

os.chdir(config.WORK)
s = np.load('split.npz', allow_pickle=True)
Xtr, ytr, Xte, yte = s['Xtr'], s['ytr'], s['Xte'], s['yte']
classes = [str(c) for c in s['classes']]
c = json.load(open(config.TUNED))
coef, names = c['coef'], c['names']
w = np.array(c['w'], np.float32)
out = {}
df = pd.DataFrame(Xte)
key = pd.util.hash_pandas_object(df, index=False).values
g = pd.DataFrame({'k': key, 'y': yte}).groupby(['k', 'y']).size().reset_index(name='n')
mx = g.groupby('k').n.max()
out['ceiling_test_rows'] = 100 * mx.sum() / len(yte)
maj = g.sort_values(['k', 'n', 'y'], ascending=[True, False, True]).drop_duplicates('k').set_index('k').y
pm = maj.loc[key].values
out['ceiling_test_per_class_recall'] = {classes[k]: round(float((pm[yte == k] == k).mean()), 4) for k in range(11)}
t = time.time()
L = layers(Xtr, ytr, Xte)
S = sum(cf * L[n] for cf, n in zip(coef, names) if cf) * w
tb = time.time() - t
srt = np.sort(S, 1)
tie = np.isclose(srt[:, -1], srt[:, -2], rtol=0, atol=1e-9)
out['ties_test'] = int(tie.sum())
out['ties_test_pct'] = 100 * float(tie.mean())
pred = S.argmax(1)
out['acc_check'] = 100 * float((pred == yte).mean())
out['ties_correct'] = int((pred[tie] == yte[tie]).sum())
out['layers_build_and_score_200k_s'] = round(tb, 1)
out['sign_test_12_of_13'] = dict(one_sided=float(binomtest(12, 13, 0.5, 'greater').pvalue),
                                 two_sided=float(binomtest(12, 13, 0.5).pvalue))
sig, sid = np.unique(Xtr[:, 1:], axis=0, return_inverse=True)
out['n_signatures'] = len(sig)
ex = np.unique(np.c_[sid, Xtr[:, 0]], axis=0)
out['n_exact_keys'] = len(ex)
z = np.arcsinh(1e6 * Xtr[:, 0])
nb = {}
for h in [0.0025, 0.01, 0.02, 0.05]:
    nb[h] = len(np.unique(np.c_[sid, np.floor(z / h)], axis=0))
out['n_bin_keys'] = {str(k): v for k, v in nb.items()}
tot = len(sig) + len(ex) + sum(nb.values())
out['total_keys'] = tot
out['bytes_counts_11x32bit'] = tot * 11 * 4
out['delta_note'] = 'z=asinh(1e6*dt); for dt>>1us, |dz|=0.05 <=> dt ratio exp(0.05)=%.4f (about 5%% relative); for dt<<1us, dz~1e6*dt so 0.05 <=> 50 ns' % np.exp(0.05)
q = np.percentile(Xtr[:, 0], [10, 50, 90])
out['dt_percentiles_s'] = q.tolist()
json.dump(out, open(os.path.join(config.out('base'), 'quick_checks.json'), 'w'), indent=1)
