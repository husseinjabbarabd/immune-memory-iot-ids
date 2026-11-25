import os
import sys

import numpy as np
from scipy.io import loadmat, savemat
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

data, wd, nj = sys.argv[1], sys.argv[2], int(sys.argv[3])
SEED = 20261124
D = loadmat(data)
Xtr, Xva, Xte = (D[k].astype(np.float32) for k in ('Xtr', 'Xva', 'Xte'))
ytr, yva = (D[k].ravel().astype(int) for k in ('ytr', 'yva'))
ben = int(D['ben'].ravel()[0])
btr, bva = (ytr != ben).astype(int), (yva != ben).astype(int)


def model(i):
    if i == 0:
        return XGBClassifier(tree_method='hist', n_jobs=nj, random_state=SEED)
    if i == 1:
        return RandomForestClassifier(n_estimators=100, min_samples_leaf=50, n_jobs=nj, random_state=SEED)
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def oof(i):
    p = np.zeros(len(btr))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xtr, btr):
        p[b] = model(i).fit(Xtr[a], btr[a]).predict_proba(Xtr[b])[:, 1]
    return p


def full(i):
    m = model(i).fit(Xtr, btr)
    return np.concatenate([m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]])


O = np.column_stack([oof(i) for i in range(3)])
F = np.column_stack([full(i) for i in range(3)])
M = RandomForestClassifier(n_estimators=100, min_samples_leaf=50, n_jobs=nj, random_state=SEED).fit(O, btr).predict_proba(F)[:, 1]
nv = len(bva)
iso = IsotonicRegression(out_of_bounds='clip').fit(M[:nv], bva)
pV, pT = iso.predict(M[:nv]), iso.predict(M[nv:])
lab = XGBClassifier(tree_method='hist', n_jobs=nj, random_state=SEED).fit(Xtr, ytr - 1).predict(Xte) + 1
savemat(os.path.join(wd, 'scores.mat'), {'pV': pV, 'pT': pT, 'lab': lab.astype(float)})
