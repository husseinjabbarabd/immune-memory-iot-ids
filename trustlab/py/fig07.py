import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

GREEN, BLUE, GREY = '#059669', '#2563EB', '#525252'
ABBR = {'API': 'API', 'Bruteforce': 'BF', 'BufferOverflow': 'BO', 'C2Beaconing': 'C2', 'DDoS': 'DDoS', 'DNS': 'DNS',
        'DoS': 'DoS', 'Evasion': 'Ev', 'Exfiltration': 'Exf', 'Exploitation': 'Expl', 'MITM': 'MITM', 'PortScan': 'PS',
        'Slowloris': 'SL', 'TLSSSL': 'TLS', 'WebBased': 'Web'}
J = json.load(open(os.path.join(config.RES, 'nested', 'nested_table.json')))
od = config.out('figures')
R = {d['family']: d for d in J['rounds']}
twins = list(config.TWINS)
fams = [d['family'] for d in J['rounds'] if d['round'] <= 15 and d['family'] not in twins]
rows = [(f, f, R[f]['det_full'], R[f]['det_control'], R[f]['det_arm']) for f in fams + twins]
rows += [(z, 'joint', R['DoS+Slowloris']['det_full_' + z], R['DoS+Slowloris']['det_control_' + z],
          R['DoS+Slowloris']['det_arm_' + z]) for z in twins]
df = pd.DataFrame(rows, columns=['family', 'round', 'full', 'control', 'arm'])
s13 = df.iloc[:13]
assert round(s13.full.mean() * 100, 2) == 85.86
assert round(s13.control.mean() * 100, 2) == 52.34
assert round(s13.arm.mean() * 100, 2) == 86.45
assert round(df.full.iloc[15] * 100, 2) == 49.64 and round(df.full.iloc[16] * 100, 2) == 36.82

plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10})
fig, ax = plt.subplots(figsize=(8.1 / 2.54, 2.3))
x = np.arange(len(df), dtype=float)
x[13:] += 0.6
x[15:] += 0.6
wb = 0.27
ax.bar(x - wb, df.arm * 100, width=wb, color=GREY, label='Arm alone at the same validation FAR')
ax.bar(x, df.full * 100, width=wb, color=GREEN, label='Base score or self-novelty arm')
ax.bar(x + wb, df.control * 100, width=wb, color='white', edgecolor=BLUE, hatch='/////', lw=0.5,
       label='Threshold-matched control')
ax.set_xticks(x, [ABBR[f] + ('*' if r == 'joint' else '') for f, r in zip(df.family, df['round'])], rotation=90)
ax.set_ylim(0, 100)
ax.set_ylabel('Detection at OP-R95 (%)')
ax.grid(axis='y')
ax.set_axisbelow(True)
for i in (13, 15):
    ax.axvline((x[i - 1] + x[i]) / 2, color='#BDBDBD', lw=0.8, ls='--')
h, l = ax.get_legend_handles_labels()
ax.legend([h[1], h[0], h[2]], [l[1], l[0], l[2]], loc='lower center', bbox_to_anchor=(0.45, 1.0), ncol=1, handlelength=1.4)
df.to_csv(os.path.join(od, 'fig07_values.csv'), index=False, float_format='%.10g')
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(od, 'fig07.' + ext), bbox_inches='tight', pad_inches=0.02, dpi=600)
