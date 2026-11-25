import hashlib
import os

import numpy as np
import scipy.io as sio

import config

FLOW_SHA = 'c91ec90b933f612aa3112f3e9082e405ff01d25a71b31b8ea2efa2553923f85f'
splits = os.path.join(config.RES, 'splits')
part = config.out('partitions')


def sha(v):
    return hashlib.sha256(np.asarray(v, np.uint8).ravel().tobytes()).hexdigest()


def write(sp, prefix):
    for code, name in ((1, 'train'), (2, 'val'), (3, 'test')):
        idx = np.flatnonzero(sp == code) + 1
        with open(os.path.join(part, prefix + name + '.txt'), 'w', newline='\n') as f:
            f.write(''.join('%d\n' % i for i in idx))


S = sio.loadmat(os.path.join(config.FS, 'FS_S01', 'FS_S01_split.mat'), variable_names=['split', 'rowId'])
sp = S['split'].ravel().astype(np.uint8)
assert np.array_equal(S['rowId'].ravel(), np.arange(1, len(sp) + 1))
assert sha(sp) == FLOW_SHA
write(sp, '')

g = np.load(os.path.join(splits, 'split_grouped.npy')).ravel()
assert np.array_equal(g, sio.loadmat(os.path.join(splits, 'split_grouped.mat'))['gsplit'].ravel())
assert len(g) == len(sp)
write(g, 'grouped_')

s = np.load(os.path.join(splits, 'split_sessions.npy')).ravel()
assert len(s) == len(sp)
write(s, 'session_')
