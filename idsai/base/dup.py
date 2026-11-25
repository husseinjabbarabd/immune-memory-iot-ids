import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.chdir(config.WORK)
s = np.load('split.npz', allow_pickle=True)
Xtr, ytr, Xte, yte, cl = s['Xtr'], s['ytr'], s['Xte'], s['yte'], s['classes']
p = np.load('final_test_eval.npz')['pred']
px = np.load('xgb_test_pred.npy')


def rows(X):
    return np.ascontiguousarray(X).view(np.dtype((np.void, X.dtype.itemsize * X.shape[1]))).ravel()


full_tr = set(rows(Xtr).tolist())
sig_tr = set(rows(np.ascontiguousarray(Xtr[:, 1:])).tolist())
ft = rows(Xte)
st = rows(np.ascontiguousarray(Xte[:, 1:]))
c1 = np.array([v in full_tr for v in ft.tolist()])
sg = np.array([v in sig_tr for v in st.tolist()])
cat = np.where(c1, 0, np.where(sg, 1, 2))
X = np.vstack([Xtr, Xte])
y = np.concatenate([ytr, yte])
r = rows(X)
u, inv = np.unique(r, return_inverse=True)
C = np.zeros((len(u), 11), np.int64)
np.add.at(C, (inv, y), 1)
ceil_all = C.max(1).sum() / len(y)
conf = (C > 0).sum(1) > 1
inv_te = inv[len(ytr):]
out = {'ceiling_multiclass_full': float(ceil_all)}
names = ['Exact 19-feature vector seen in training', 'Known signature, unseen Δt', 'Unseen signature']
for k in range(3):
    m = cat == k
    out[names[k]] = {'n': int(m.sum()), 'frac': float(m.mean()), 'ais_acc': float((p[m] == yte[m]).mean()),
                     'frac_label_conflicted': float(conf[inv_te[m]].mean()),
                     'xgb_acc': float((px[m] == yte[m]).mean())}
out['exact_repeat_frac_per_class'] = {str(cl[c]): float(c1[yte == c].mean()) for c in range(11)}
ut, ct = np.unique(rows(Xtr), return_counts=True)
out['train_unique_vectors'] = int(len(ut))
out['train_rows'] = int(len(Xtr))
json.dump(out, open(os.path.join(config.out('base'), 'dup.json'), 'w'), indent=1)
