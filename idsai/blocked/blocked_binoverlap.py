import json
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.chdir(config.WORK)
y = np.load('idsai19.npz')['y'].astype(str)
n = len(y)
S = np.load('split.npz', allow_pickle=True)
te_pub = S['ite']
te_blk = np.load('blocked_preds.npz')['te']
yb = (y != 'Normal').astype(int)
out = {}
for name, args in [('indices', (np.arange(n),)), ('indices_y', (np.arange(n), yb))]:
    r = train_test_split(*args, test_size=0.2, stratify=yb, random_state=21)
    itr = r[0]
    out[name] = dict(published_test_in_binary_train=int(np.isin(te_pub, itr).sum()),
                     blocked_test_in_binary_train=int(np.isin(te_blk, itr).sum()))
out['check_value'] = 159919
out['blocked_share'] = {k: 100 * v['blocked_test_in_binary_train'] / len(te_blk) for k, v in out.items() if isinstance(v, dict)}
json.dump(out, open(os.path.join(config.out('blocked'), 'blocked_binoverlap.json'), 'w'), indent=1)
