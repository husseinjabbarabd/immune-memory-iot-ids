import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_p = json.load(open(os.path.join(ROOT, 'paths.json')))['idsai']

CSV = os.path.normpath(os.path.join(ROOT, _p['csv']))
WORK = os.path.normpath(os.path.join(ROOT, _p['work']))
RES = os.path.normpath(os.path.join(ROOT, _p['results']))
TUNED = os.path.join(RES, 'base', 'ais_tuned.json')
RULE = os.path.join(RES, 'base', 'zeroday_rule.json')
PROSPECTIVE = os.path.join(RES, 'base', 'prospective_fpr.json')
TABLE10 = os.path.join(ROOT, 'reference', 'table10_5pct.csv')

sys.path.insert(0, os.path.join(HERE, 'base'))


def out(name):
    d = os.path.join(RES, name)
    os.makedirs(d, exist_ok=True)
    return d
