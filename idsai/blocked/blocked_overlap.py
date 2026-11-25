import json
import os
import sys

import numpy as np
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.chdir(config.WORK)
d = np.load('idsai19.npz')
X = d['X'].astype(np.float64)
le = LabelEncoder()
y = le.fit_transform(d['y'])
cl = [str(c) for c in le.classes_]
K = len(cl)
S = np.load('split.npz', allow_pickle=True)
te = np.load('blocked_preds.npz')['te']
assert np.array_equal(S['yte'], y[S['ite']]) and np.array_equal(S['Xte'].astype(np.float64), X[S['ite']])
tr = np.setdiff1d(np.arange(len(y)), te)


def rows(A):
    return np.ascontiguousarray(A).view(np.dtype((np.void, A.dtype.itemsize * A.shape[1]))).ravel()


un = ~np.isin(rows(X[te][:, 1:]), rows(X[tr][:, 1:]))
out = dict(blocked_test_in_published_train=int(np.isin(te, S['itr']).sum()),
           share=100 * float(np.isin(te, S['itr']).mean()),
           unseen_sig_true_class={cl[k]: int(v) for k, v in enumerate(np.bincount(y[te][un], minlength=K)) if v})
json.dump(out, open(os.path.join(config.out('blocked'), 'blocked_overlap.json'), 'w'), indent=1)
