import csv
import hashlib
import json
import os

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
p = json.load(open(os.path.join(root, 'paths.json')))['trustlab']
src = os.path.join(root, os.path.dirname(p['raw']), 'PortScan.csv')
dst = os.path.join(root, p['raw'], 'PortScan_recovered.csv')
res = os.path.join(root, p['results'])
os.makedirs(res, exist_ok=True)
lst = os.path.join(res, 'portscan_removed_rows.csv')

SRC = 'fa349c5fe0c4236233873d7feb03953c7735b184d9c69b83732d3a5b4559003f'
DST = 'c7015f31621c1676c7314f3d2fc87a56f3ffab3adba6c7d61e6a9a8bf93b27ca'
KEEP = 107566

b = open(src, 'rb').read()
if hashlib.sha256(b).hexdigest() != SRC:
    raise SystemExit('PortScan.csv is not the released file')
lines = b.split(b'\r\n')
out = b'\r\n'.join(lines[:KEEP + 1])
if hashlib.sha256(out).hexdigest() != DST:
    raise SystemExit('output differs from PortScan_recovered.csv')
open(dst, 'wb').write(out)

with open(lst, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['line', 'data_row', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 'protocol', 'timestamp', 'fields', 'status'])
    for i in range(KEEP + 1, len(lines)):
        v = lines[i].decode().split(',')
        st = 'complete row' if len(v) == 79 else 'truncated line (%d of 79 fields, no label)' % len(v)
        w.writerow([i + 1, i] + v[:6] + [len(v), st])
