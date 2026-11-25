# Immune-inspired memory detectors for IoT intrusion detection

Code for the paper "Immune-Inspired Memory Detectors for IoT Intrusion Detection: An Empirical Label-Conflict Ceiling and Zero-Day Evaluation", International Journal of Intelligent Engineering and Systems (IJIES). It runs the experiments of the paper on the two public datasets, IDSAI and TRUSTLab, from the raw files to the tables and figures.

## Requirements

- Python 3.11 (we used 3.11.9) with the packages in requirements.txt: `pip install --ignore-requires-python --only-binary=:all: -r requirements.txt`. The xgboost 3.4.1 wheel asks for Python 3.12, but our runs used it on 3.11.9; the first option lets pip install it there.
- MATLAB R2025b with the Parallel Computing Toolbox and the Statistics and Machine Learning Toolbox
- 16 GB of RAM and about 10 GB of free disk space

## Data

Neither dataset is included. Download them from their official pages and place them as below (the paths can be changed in paths.json):

- IDSAI, https://github.com/BioAITeam/Intrusion-Detection-System-using-Machine-Learning: IDSAI.csv in data/idsai/
- TRUSTLab, https://doi.org/10.82432/10317/21203: the class files (API.csv, Benign.csv, ...) in data/trustlab/01_Data_Raw/ and PortScan.csv in data/trustlab/

If a file is missing, run_all stops and names the path it expects.

## Running

    python idsai/run_all.py

and, in MATLAB, from the trustlab folder:

    run_all

`run_all('published')` takes the feature selections stored in reference/ instead of running the clonal selection again. MATLAB calls Python for some steps; set the environment variable PYTHON if `python` is not on the path. Results are written to results/; the IDSAI scripts also keep intermediate arrays (split, test predictions) in work/idsai/.

## Outputs

Table and figure numbers are those of the paper. Some script names keep the numbers of an earlier draft: idsai/base/fig04.py draws Fig. 3 and fig05.py the per-fold plot; trustlab/py/fig07.py, fig08.py and fig09.py draw Figs. 5, 6 and 7.

IDSAI (results/idsai/)
- base: Tables 4, 5, 6 and 8, the values of Fig. 3 and the per-fold accuracies of Table 5; model.npz is the trained detector (the exact-match and bin-width count tables, the layer weights and the class weights), and it is checked against the test predictions when it is written
- figures: Fig. 3 and a plot of the per-fold accuracies, drawn from the base values
- base and conflict: Table 7
- blocked: the blocked-split results (Section 5.2)
- nested: Tables 9 and 10 (tables_paper.txt; its panel labels are those of our working draft) and Fig. 4 (fig_curves); compare_5pct.py compares the 5% results with reference/table10_5pct.csv
- fpr: the comparison with frozen coefficients of the submitted version (Table 20, submitted column); fpr_rows.py prints it under the headings TABLE 9 and TABLE 10, the table numbers of the submitted version
- partitions: the rows of the published split (train, calibration, test) and of the blocked split

TRUSTLab (results/trustlab/)
- data: Table 2(b)
- pipeline: the closed-set model, Tables 13 and 14, the values of Figs. 2, 6 and 7, and the submitted zero-day protocol (Section 5.6); model.mat is the trained model (coding of the selected features, feature and specialist weights, AIRS2 cells, thresholds and exact memory)
- study: Tables 13, 14 and 15
- nested: Tables 11 and 12, Fig. 5 and the per-round settings (Section 5.4)
- components: Table 16 and the negative-selection and k-means columns of Table 11
- grouped and splits: Tables 18 and 19
- portfree and common_far: Tables 17 and 18 and the port-free results of Section 5.6
- figures: Figs. 5, 6 and 7, and fig7b.png (held-out detection against false-alarm rate, with and without the ports)
- exploit and nu_diag: the Exploitation round (Section 5.4) and the self-novelty threshold on the time-grouped split (Section 6.5)
- partitions: the rows of the flow-level, time-grouped and session splits; the session split is written by py/session_split.py for a check in Section 6.5, and no reported result uses it

XGBoost results can differ in the last decimals between machines, as reported in Section 5.1 of the paper.
The per-packet predictions of the original tuned-XGBoost run were not kept, so circ_block.py reproduces the rerun's block-bootstrap intervals given in Section 5.1, not the recorded ones.

Table 20 repeats values from these tables.

## Reference files

reference/ holds outputs of our own runs; the TRUSTLab pipeline compares its results with them in pipeline/checks.csv, and reference/SHA256SUMS.txt gives their hashes. The hashes are of the files as stored in the repository, with LF line endings: a release zip matches them, while a Windows clone with core.autocrlf=true writes the CSV files with CRLF and does not. Up to v1.2 the list gave the CRLF hashes of the 24 CSV files outside per_row/; the files themselves are unchanged.
- idsai_model.npz: the trained IDSAI detector
- portscan_removed_rows.csv: the 13 PortScan rows dropped when the file was recovered
- round_outputs.csv: the feature selection, weights and settings of each of the 16 nested TRUSTLab rounds; run_nested reads it when run_all('published') is used
- table10_5pct.csv: the 5% results of the submitted version (family, ADR, EUR), read by idsai/nested/compare_5pct.py
- 03_Outputs_FS: the stored TRUSTLab closed-set run. FS_S03: clonal selection at k = 10 to 74 and the chosen selection; FS_S04: the expanded (specialist) weights; FS_S05: the AIRS2 pool and the cells of each class; FS_S06: validation state and thresholds; FS_S07: test and validation results and state
- 03_Outputs_ZD: the stored rounds of the submitted zero-day protocol. ZD_S01: base table and the classes assigned by the first phase; ZD_S02: base score against self-novelty; ZD_S03: rules at equal false-alarm rate

partitions/ lists the rows of every split.

## Per-row outputs

reference/per_row/ holds per-row test outputs as gzip CSV. Flags are 1 when the flow is flagged; scores are printed to nine significant digits and the flags were computed from the full-precision scores.
- idsai_test_predictions.csv.gz: the 200,000 IDSAI test packets; row (IDSAI.csv data row, from 0), label, predicted (Tables 4 and 7, Fig. 3)
- idsai_test_baselines.csv.gz: the same 200,000 packets; row, label, and the class predicted by each baseline: xgb_tuned_rerun (92.5615%, the rerun on our second computer used in the tuned column of Table 7; Table 4 keeps the recorded 92.565%), xgb_default (92.534%, Table 4), xgb_table3 (92.524%, the settings of Table 3 of the original study, Section 5.1), knn_k51 (92.556% on the second computer; Table 4 keeps the recorded 92.561%), knn_k201 (92.50%, Section 3.6), lookup (92.178%, Table 4) and mlp (the final MLP of base/mlp_final.py, H = 5, 87.418%)
- idsai_blocked_predictions.csv.gz: the 200,000 test packets of the blocked split (Section 5.2); row (as in partitions/idsai/blocked_test.txt), label, and the class predicted by immune (91.1675%), xgb_tuned (91.0785%), xgb_default (90.920%), xgb_retuned (90.918%), knn (89.251%) and lookup (90.2705%)
- trustlab_test_outputs.csv.gz: the 900,334 TRUSTLab test flows; row (from 1, as in partitions/trustlab/test.txt), label, score (first phase), and the label at OP-SPEC, OP-F2 and OP-R95 (Tables 13 and 14, Figs. 6 and 7); a flow is forwarded to the second phase when its score is at least the threshold of the operating point (0.200124934, 0.406930029 and 0.555784523), which forwards 476,146, 402,894 and 370,845 flows
- nested_r01.csv.gz to nested_r16.csv.gz: the 900,334 TRUSTLab test flows in each of the 16 nested held-out rounds (Table 11, Fig. 5). Rounds 1 to 15 withhold API, Bruteforce, BufferOverflow, C2Beaconing, DDoS, DNS, DoS, Evasion, Exfiltration, Exploitation, MITM, PortScan, Slowloris, TLSSSL and WebBased; round 16 withholds DoS and Slowloris together. Columns: row, label, round, s (base score), q (self-novelty score), and the flag of the combined rule (comb), the base score alone (base), the control at the same validation false-alarm rate (ctrl) and the self-novelty arm alone (arm), each at OP-SPEC, OP-F2 and OP-R95 (spec, f2, r95)
- nested_thresholds.csv: the thresholds of each nested round and operating point (tau and nu of the combined rule, tauM of the control, nuM of the arm)
- portfree_nested_r01.csv.gz to portfree_nested_r16.csv.gz and portfree_nested_thresholds.csv: the same 16 rounds without the port features (Table 17, Section 5.6), with the columns of the nested files
- grouped_submitted_zd01.csv.gz and grouped_submitted_zd03.csv.gz to grouped_submitted_zd16.csv.gz, and grouped_submitted_thresholds.csv: the 944,917 test flows of the time-grouped split in the 15 rounds of the submitted zero-day protocol (Section 5.6, Table 20); zdNN withholds class NN in the order API, Benign, Bruteforce, BufferOverflow, C2Beaconing, DDoS, DNS, DoS, Evasion, Exfiltration, Exploitation, MITM, PortScan, Slowloris, TLSSSL, WebBased (Benign is never withheld, so there is no zd02); row as in partitions/trustlab/grouped_test.txt, and the columns of the nested files
- portfree_closed_test.csv.gz and grouped_closed_test.csv.gz, with portfree_closed_thresholds.csv and grouped_closed_thresholds.csv: the closed-set model without the port features (900,334 flows, row as in partitions/trustlab/test.txt) and on the time-grouped split (944,917 flows, row as in partitions/trustlab/grouped_test.txt) (Table 18, Section 5.6). Columns: row, label, score (first phase), q (self-novelty score), the label at OP-SPEC, OP-F2 and OP-R95 (op_spec, op_f2, op_r95), and 1 when the flow is forwarded to the second phase at each operating point (fwd_spec, fwd_f2, fwd_r95); the thresholds files give tau for each operating point

A few flows score less than 1e-16 below a threshold and print the same as it to nine digits; a flag recomputed from the printed values marks them, while the files and the paper do not:
- nested round 4, OP-F2: one flow (row 3852893)
- port-free nested round 6, control at OP-R95: 5,587 flows that share one score, 5.6e-17 below tauM
- port-free nested round 9, control at OP-SPEC: one flow (row 1784034)
- port-free closed set, OP-SPEC: one flow (row 1310644)

The other per-row outputs are not in the release. The scripts write them when they are run:
- TRUSTLab submitted zero-day protocol on the random split with the ports, run_pipeline.m: pipeline/zdNN/score_test.mat and zd_NN.mat
- TRUSTLab submitted zero-day protocol on the random split without the ports, run_portfree.m: portfree/zdNN/score_test.mat and zd_NN.mat

The negative-selection and k-means rules of the TRUSTLab open-set comparison (run_open.m) write per-round rates only (components/open_nested.csv). The IDSAI held-out (zero-day) rounds write per-round counts and rates, the values reported in the paper; their per-packet outputs were not stored.

## Settings and seeds

idsai/config.py holds only the paths. The IDSAI immune-detector grids are in idsai/base/ais.py (and zd_nested.py, blocked/blocked.py), the XGBoost grids in nested/nested_fpr.py, nested/nested_xgb_def.py and blocked/blocked_xgb.py, and the tuned XGBoost settings in base/xgb_tuned.py; the tuned immune settings and the zero-day rule are written to results/idsai/base/ais_tuned.json and zeroday_rule.json. IDSAI seeds: 21 for the published split, the calibration split and the subsamples; XGBoost random_state 0; bootstrap draws 7 and 1; the MLP seeds are set in base/mlp_sweep.py and mlp_final.py.

TRUSTLab settings, with the clonal-selection and AIRS2 parameters, are in trustlab/tl_cfg.m. TRUSTLab seeds: split 20261020, sampling frame 20261012, clonal selection 20261100, specialist 20261204, AIRS2 20261010, mutation 20268010. The remaining seeds are fixed in the scripts that use them.

## Run time

On one PC (Intel i7-11800H, 16 GB RAM): IDSAI about one to two days, mostly the nested zero-day comparison; TRUSTLab about five to seven days with two MATLAB workers, mostly the held-out-family rounds.

## License

MIT (LICENSE).
