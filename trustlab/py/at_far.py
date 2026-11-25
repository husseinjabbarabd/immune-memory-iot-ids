import csv
import os

import config

D = os.path.join(config.RES, 'common_far')
X = 0.01169

cur = {}
for r in csv.DictReader(open(os.path.join(D, 'curves.csv'))):
    cur.setdefault((r['version'], r['detector']), []).append((float(r['target_val_far']), float(r['mean_test_far']), float(r['mean_detection'])))

rows, lines = [], ['mean held-out detection at mean test false-alarm rate %.3f%% (linear between bracketing grid points)' % (100 * X), '']
for ver in ('port-using', 'port-free'):
    for det in ('full rule', 'control', 'arm alone'):
        P = sorted(cur[(ver, det)])
        assert all(P[i][1] <= P[i + 1][1] for i in range(len(P) - 1)), (ver, det)
        i = next(i for i in range(len(P) - 1) if P[i][1] <= X <= P[i + 1][1])
        a, b = P[i], P[i + 1]
        y = a[2] if b[1] == a[1] else a[2] + (X - a[1]) / (b[1] - a[1]) * (b[2] - a[2])
        rows.append([ver, det, '%.15g' % y, '%.3f' % a[0], '%.15g' % a[1], '%.15g' % a[2], '%.3f' % b[0], '%.15g' % b[1], '%.15g' % b[2]])
        lines.append('%-11s %-10s %8.4f   between g=%.3f (FAR %.5f%%, ADR %.4f) and g=%.3f (FAR %.5f%%, ADR %.4f)'
                     % (ver, det, 100 * y, a[0], 100 * a[1], 100 * a[2], b[0], 100 * b[1], 100 * b[2]))

with open(os.path.join(D, 'at_1169.csv'), 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['version', 'detector', 'detection', 'g_lo', 'far_lo', 'det_lo', 'g_hi', 'far_hi', 'det_hi'])
    w.writerows(rows)
open(os.path.join(D, 'at_1169.txt'), 'w').write('\n'.join(lines) + '\n')
