import json
import os

import config

R = config.out('study')
o = json.load(open(os.path.join(R, 'op_metrics.json')))
a = json.load(open(os.path.join(R, 'study_audit.json')))['ours']
wb, wa = a['w_benign'], a['w_attack']
out = {}
for op in ('OP-SPEC', 'OP-F2', 'OP-R95'):
    c = o[op]
    tp, fp, fn, tn = c['TP'] * wa, c['FP'] * wb, c['FN'] * wa, c['TN'] * wb
    p, r = tp / (tp + fp), tp / (tp + fn)
    out[op] = {'acc_pc': (tp + tn) / (tp + fp + fn + tn), 'prec_pc': p, 'f2_pc': 5 * p * r / (4 * p + r),
               'rec': r, 'spec': tn / (tn + fp)}
json.dump(out, open(os.path.join(R, 'reweight.json'), 'w'), indent=1)
