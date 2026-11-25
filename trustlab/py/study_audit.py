import csv
import json
import os

import numpy as np

import config

FS = os.path.join(config.REF, 'FS_S07')
OUT = config.out('study')
cls = ['API', 'Benign', 'Bruteforce', 'BufferOverflow', 'C2Beaconing', 'DDoS', 'DNS', 'DoS', 'Evasion',
       'Exfiltration', 'Exploitation', 'MITM', 'PortScan', 'Slowloris', 'TLSSSL', 'WebBased']
B = 1

F14 = np.array([
    [41395, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 431824, 1, 1790, 0, 0, 0, 0, 0, 0, 2708, 1, 0, 0, 0, 8],
    [0, 3, 26889, 0, 0, 0, 0, 0, 0, 2, 549, 0, 0, 0, 0, 1],
    [0, 742, 0, 35275, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
    [0, 3, 0, 0, 34469, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 56645, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 50429, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 31966, 0, 0, 0, 0, 0, 13000, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 25311, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 32318, 0, 0, 0, 0, 0, 0],
    [0, 1990, 970, 142, 0, 0, 0, 0, 0, 2, 39508, 2, 0, 0, 26, 198],
    [0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 31818, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 452, 0, 0, 0, 0, 0, 0, 41294, 0, 0, 12],
    [0, 0, 0, 0, 0, 0, 0, 9310, 0, 0, 0, 0, 0, 23325, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 35066, 0],
    [0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 149, 0, 33, 0, 0, 34531]])
NB, NA, N = 436332, 597862, 1034194
TP, FN, TN, FP = 567844, 30018, 359053, 77279
FWD, BACK = 645123, 75515
T3 = dict(API=41395, Bruteforce=27444, BufferOverflow=36019, C2Beaconing=34472, DDoS=56645, DNS=50429, DoS=44966,
          Evasion=25312, Exfiltration=32318, Exploitation=42838, MITM=31821, PortScan=41758, Slowloris=32635,
          TLSSSL=35073, WebBased=34719)


def prf(M, rows, cols_macro):
    R = M[rows]
    P = np.zeros(len(cls))
    Rc = np.zeros(len(cls))
    F = np.zeros(len(cls))
    for k in range(len(cls)):
        tp = R[:, k][rows.index(k)] if k in rows else 0
        cs = R[:, k].sum()
        rs = M[k].sum() if k in rows else 0
        P[k] = tp / cs if cs else 0
        Rc[k] = tp / rs if rs else 0
        F[k] = 2 * P[k] * Rc[k] / (P[k] + Rc[k]) if P[k] + Rc[k] else 0
    acc = sum(M[k, k] for k in rows) / R.sum()
    return acc, F[cols_macro].mean(), P, Rc, F


att = [k for k in range(16) if k != B]
chk = {}
chk['attack rows = Table 3 supports'] = all(int(F14[k].sum()) == T3[cls[k]] for k in att)
chk['attack row total = TP 567,844'] = int(F14[att].sum()) == TP
chk['Benign row total = all benign test flows 436,332'] = int(F14[B].sum()) == NB
fa = int(F14[B].sum() - F14[B, B])
G = F14.copy()
G[B, B] = FP - fa
chk['predicted Benign among forwarded = 75,515 (text p. 16)'] = int(G[:, B].sum()) == BACK
chk['forwarded total = 645,123'] = int(G.sum()) == FWD

fA, fM, _, _, fF = prf(G, list(range(16)), list(range(16)))
aA, aM, aP, aR, aF = prf(G, att, att)
e2e_exact = (TN + sum(G[k, k] for k in range(16))) / N
e2e_formula = (TN + round(fA, 4) * FWD) / N
Fb = []
for k in att:
    p = G[k, k] / G[:, k].sum()
    r = G[k, k] / G[k].sum()
    Fb.append(2 * p * r / (p + r))
rec = {
    'fwd16_acc': fA, 'fwd16_mf1': fM, 'att15_acc': aA, 'att15_mf1': aM, 'att15_mf1_with_benign_fp': float(np.mean(Fb)),
    'e2e_acc_exact_counts': e2e_exact, 'e2e_acc_printed_formula': e2e_formula,
    'false_alarms_fig14': fa, 'false_alarm_share_fig14': fa / N, 'false_alarm_share_text': '~2.2%',
    'checks': chk}

cm = []
with open(os.path.join(FS, 'FS_S07_TEST_confusion_R95.csv')) as f:
    r = csv.reader(f)
    next(r)
    for row in r:
        cm.append([int(x) for x in row[1:]])
cm = np.array(cm)
n = cm.sum(1)
nb = int(n[B])
na = int(n.sum() - nb)
cb = int(cm[B, B])
ca = int(sum(cm[k, k] for k in att))
pc = (NB * cb / nb + NA * ca / na) / N
own = (cb + ca) / (nb + na)
res = list(csv.DictReader(open(os.path.join(FS, 'FS_S07_TEST_results.csv'))))
ours = {'n_test': int(n.sum()), 'n_benign': nb, 'n_attack': na, 'per_class': {c: int(x) for c, x in zip(cls, n)},
        'R95_correct_benign': cb, 'R95_correct_attack': ca, 'e2e_own': own, 'e2e_pc': pc,
        'e2e_pc_saved': float(res[2]['e2e_Acc_pc']), 'w_benign': NB / nb, 'w_attack': NA / na}
json.dump({'study': rec, 'ours': ours, 'study_att15_per_class': {cls[k]: [aP[k], aR[k], aF[k]] for k in att}},
          open(os.path.join(OUT, 'study_audit.json'), 'w'), indent=1, default=float)
