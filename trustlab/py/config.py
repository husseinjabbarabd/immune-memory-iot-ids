import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_p = json.load(open(os.path.join(ROOT, 'paths.json')))['trustlab']

FS = os.path.join(ROOT, _p['fs'])
OUTS = os.path.join(ROOT, _p['out'])
REF = os.path.join(ROOT, 'reference', '03_Outputs_FS')
REFZD = os.path.join(ROOT, 'reference', '03_Outputs_ZD')
COUNTS = os.path.join(ROOT, _p['counts'])
RES = os.path.join(ROOT, _p['results'])
S09 = os.path.join(FS, 'FS_S00', 'S09_features.mat')
NAMES = os.path.join(FS, 'FS_S00', 'SUP_class_names.mat')
CLASSES = ['API', 'Benign', 'Bruteforce', 'BufferOverflow', 'C2Beaconing', 'DDoS', 'DNS', 'DoS', 'Evasion',
           'Exfiltration', 'Exploitation', 'MITM', 'PortScan', 'Slowloris', 'TLSSSL', 'WebBased']
TWINS = ('DoS', 'Slowloris')


def out(name):
    d = os.path.join(RES, name)
    os.makedirs(d, exist_ok=True)
    return d
