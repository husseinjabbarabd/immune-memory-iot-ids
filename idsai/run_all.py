import os
import subprocess
import sys

import config

STEPS = '''
base/build_npz base/prep
base/ais base/final_eval base/model base/cv10
base/xgb_runs base/xgb_tuned base/xgb_nj base/xgbT_cv base/xgb_table3 base/knn base/knn2 base/lookup
base/mlp_sweep base/mlp_asinh base/mlp_asinh2 base/mlp_final
base/stats base/stats_tuned base/mcnemar_baselines base/dup base/cv_paired base/circ_block base/quick
base/ablation base/sens base/quick_checks base/fig04 base/fig05
base/zeroday_strict base/zeroday_tune base/zeroday_final base/zd_baselines base/xgb_zeroday base/zd_nested
base/prospective_fpr
fpr/fpr_table fpr/fpr_rows fpr/xgb_ties
nested/nested_fpr nested/nested_wn nested/path_conflict nested/nested_xgb_def nested/xgb_def_checks
nested/leak_audit nested/leak_audit_xgbdef nested/leak_disjoint nested/curves_oneclass nested/fig_curves
nested/compare_5pct nested/tables_nested nested/tables_paper
conflict/conflict_decomp conflict/conflict_share_check conflict/table7_tuned_col
blocked/blocked blocked/blocked_xgb blocked/blocked_knn blocked/blocked_extra blocked/blocked_overlap
blocked/blocked_binoverlap blocked/blocked_perclass
base/partitions
'''.split()

if not os.path.isfile(config.CSV):
    sys.exit('IDSAI data not found: %s. Get IDSAI.csv from '
             'https://github.com/BioAITeam/Intrusion-Detection-System-using-Machine-Learning and put it there.' % config.CSV)
here = os.path.dirname(os.path.abspath(__file__))
for s in STEPS:
    print(s, flush=True)
    subprocess.run([sys.executable, os.path.join(here, s + '.py')], check=True)
