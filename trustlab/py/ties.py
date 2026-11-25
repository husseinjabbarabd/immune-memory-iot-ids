import os

import numpy as np
from scipy.io import loadmat

import config
from common_far import NAMES, TARGET, TWINS, roundk, sq, full_tau

wd = os.path.join(config.RES, 'portfree')
out, rates = [], []
for fam in [n for n in NAMES if n != 'Benign' and n not in TWINS]:
    x = NAMES.index(fam) + 1
    nu = float(loadmat(os.path.join(wd, 'zd_%02d.mat' % x), variable_names=['nu'])['nu'].ravel()[0])
    sv, qv, _ = sq(os.path.join(wd, 'zd%02d' % x, 'score_val.mat'), 10)
    base = qv >= nu
    K = roundk(TARGET * sv.size) - int(base.sum())
    r = sv[~base]
    sd = np.sort(r)[::-1]
    v = sd[K - 1]
    tie = int(np.sum(r == v))
    above = int(np.sum(r > v))
    flagged = above + int(base.sum())
    t = full_tau(sv, qv, nu, TARGET)
    rule = int(base.sum()) + (0 if t is None else int(np.sum(r >= t)))
    out.append('%-13s n=%d target=%d arm=%d K=%d score at K=%.6f tied rows=%d rows above=%d -> without the tied block %d (%.3f%%); '
               'with it %d (%.3f%%); matching rule flags %d (%.3f%%)' % (fam, sv.size, roundk(TARGET * sv.size), int(base.sum()), K, v, tie,
               above, flagged, 100 * flagged / sv.size, flagged + tie, 100 * (flagged + tie) / sv.size, rule, 100 * rule / sv.size))
    rates.append(rule / sv.size)
out.append('mean validation rate over %d rounds: %.5f%%' % (len(rates), 100 * np.mean(rates)))
open(os.path.join(config.out('common_far'), 'ties.txt'), 'w').write('\n'.join(out) + '\n')
