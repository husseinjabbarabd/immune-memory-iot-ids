import csv
import json
import os

import config

src = config.OUTS
od = config.out('data')
man = {r['file']: r for r in csv.DictReader(open(os.path.join(src, 'S01', 'S01_manifest.csv')))}
lab = {}
for r in csv.DictReader(open(os.path.join(src, 'S01', 'S01_labels.csv'))):
    lab.setdefault(r['file'], []).append('%s (%s)' % (r['raw_label'], format(int(r['count']), ',')))
kept = {r['file'].replace('_recovered', ''): int(r['rows_read']) for r in csv.DictReader(open(os.path.join(src, 'S09', 'S09_file_rows.csv')))}
rows = []
for f in sorted(man):
    n = int(man[f]['data_rows'])
    k = kept[f]
    rows.append({'class': f[:-4], 'labels': lab[f], 'complete': n, 'retained': k, 'removed': n - k,
                 'cols': int(man[f]['n_cols']), 'raw_lines': int(man[f]['raw_lines'])})
tot = {k: sum(r[k] for r in rows) for k in ('complete', 'retained', 'removed')}
assert tot['complete'] == 4501691 and tot['retained'] == 4501678 and tot['removed'] == 13
assert [r['class'] for r in rows if r['removed']] == ['PortScan']
sch = {}
for r in csv.DictReader(open(os.path.join(src, 'S01', 'S01_schema.csv'))):
    sch.setdefault(r['file'], []).append(r['col_name'])
allc = sorted({c for v in sch.values() for c in v})
every = [c for c in allc if all(c in v for v in sch.values())]
notall = {c: sorted(f[:-4] for f, v in sch.items() if c in v) for c in allc if c not in every}
ids = ['Src IP', 'Dst IP', 'Timestamp', 'Label']
ports = ['Src Port', 'Dst Port', 'Protocol']
flow = [c for c in sch['API.csv'] if c in every and c not in ids + ports]
assert len(flow) == 71, len(flow)
json.dump({'rows': rows, 'total': tot, 'flow_features': flow, 'ports': ports, 'not_in_every_file': notall,
           'identifiers': ids}, open(os.path.join(od, 'retained_flows.json'), 'w'), indent=1)
L = ['%s | %s | %s | %s | %d' % (r['class'], '; '.join(r['labels']), format(r['complete'], ','), format(r['retained'], ','), r['removed']) for r in rows]
L += ['total %s' % tot, 'flow features (71): ' + ', '.join(flow), 'not in every file: %s' % notall]
open(os.path.join(od, 'retained_flows.txt'), 'w').write('\n'.join(L) + '\n')
