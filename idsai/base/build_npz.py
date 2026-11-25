import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.makedirs(config.WORK, exist_ok=True)
df = pd.read_csv(config.CSV)
f = df.drop(['label', 'tipo_ataque'], axis=1).drop(['ip_src', 'ip_dst', 'port_src', 'port_dst', 'protocols'], axis=1)
X = f.values.astype(np.float64)
y = np.array(df['tipo_ataque'].tolist(), dtype=str)
feats = np.array([str(c) for c in f.columns])
np.savez(os.path.join(config.WORK, 'idsai19.npz'), X=X, y=y, feats=feats)
out = dict(rows=int(len(X)), cols=int(X.shape[1]), nan=int(np.isnan(X).sum()),
           classes={k: int(v) for k, v in zip(*np.unique(y, return_counts=True))})
json.dump(out, open(os.path.join(config.out('base'), 'build_npz.json'), 'w'), indent=1)
