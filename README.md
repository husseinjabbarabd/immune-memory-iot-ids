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

`run_all('published')` takes the feature selections stored in reference/ instead of running the clonal selection again; the results are the same. MATLAB calls Python for some steps; set the environment variable PYTHON if `python` is not on the path. Everything is written to results/.

## Outputs

IDSAI (results/idsai/)
- base: Tables 4, 5, 6 and 8, the values of Figs. 4 and 5; model.npz is the trained detector (the exact-match and bin-width count tables, the layer weights and the class weights), and it is checked against the test predictions when it is written
- figures: Figs. 4 and 5, drawn from the base values
- base and conflict: Table 7
- blocked: Table 7a
- base and fpr: Table 9
- nested: Table 10 and Fig. 6
- partitions: the rows of the published split (train, calibration, test) and of the blocked split

TRUSTLab (results/trustlab/)
- data: Table 2(c)
- pipeline: the closed-set model, Tables 12 and 13, the values of Figs. 3, 8 and 9, and the submitted zero-day protocol; model.mat is the trained model (coding of the selected features, feature and specialist weights, AIRS2 cells, thresholds and exact memory)
- study: Tables 12, 13 and 14
- nested: Tables 11, 11b, 11c and 11d
- components: Tables 11e and 12b
- grouped and splits: Tables 11f and 12c
- portfree and common_far: Tables 11g, 12c, 12d and 12e
- figures: Figs. 7, 7b, 8 and 9
- exploit and nu_diag: the Exploitation round (Section 5.4) and the self-novelty threshold on the time-grouped split (Section 6.5)
- partitions: the rows of the flow-level, time-grouped and session splits

XGBoost results can differ in the last decimals between machines, as reported in Section 5.1 of the paper.
The per-packet predictions of the original tuned-XGBoost run were not kept, so circ_block.py reproduces the rerun's block-bootstrap intervals given in Section 5.1, not the recorded ones.

Table 15 repeats values from these tables. reference/ holds outputs of our own runs (the trained IDSAI detector idsai_model.npz, the TRUSTLab feature selections and weights, AIRS2 cells, thresholds, per-round settings, and the 13 PortScan rows dropped when the file was recovered); the pipeline compares its results with them in pipeline/checks.csv. partitions/ lists the rows of every split; reference/SHA256SUMS.txt gives the file hashes.

## Settings and seeds

Settings are in idsai/config.py, the idsai scripts and trustlab/tl_cfg.m. The IDSAI split is the published one (test share 0.2, stratified, random state 21). TRUSTLab seeds: split 20261020, sampling frame 20261012, clonal selection 20261100, specialist 20261204, AIRS2 20261010, mutation 20268010. The remaining seeds are fixed in the scripts that use them.

## Run time

On one PC (Intel i7-11800H, 16 GB RAM): IDSAI about one to two days, mostly the nested zero-day comparison; TRUSTLab about five to seven days with two MATLAB workers, mostly the held-out-family rounds.

## License

MIT (LICENSE).
