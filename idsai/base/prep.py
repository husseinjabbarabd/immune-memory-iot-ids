import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.chdir(config.WORK)
d = np.load('idsai19.npz')
X = d['X'].astype(np.float64)
ys = d['y']
feats = list(d['feats'])
features = pd.DataFrame(X, columns=feats)
idx = np.arange(len(X))
Xtr, Xte, ytr, yte, itr, ite = train_test_split(features, ys, idx, test_size=0.2, random_state=21, stratify=ys)
le = LabelEncoder()
ytr = le.fit_transform(ytr)
yte = le.transform(yte)
np.savez('split.npz', Xtr=Xtr.values, Xte=Xte.values, ytr=ytr, yte=yte, itr=itr, ite=ite, classes=le.classes_,
         feats=np.array(feats))
