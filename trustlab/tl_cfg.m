function c = tl_cfg()
root = fileparts(fileparts(mfilename('fullpath')));
j = jsondecode(fileread(fullfile(root, 'paths.json'))); p = j.trustlab;
c.src   = fullfile(root, p.fs);
c.s09   = fullfile(c.src, 'FS_S00', 'S09_features.mat');
c.sup   = fullfile(c.src, 'FS_S00', 'SUP_S01_split.mat');
c.names = fullfile(c.src, 'FS_S00', 'SUP_class_names.mat');
c.ref   = fullfile(root, 'reference', '03_Outputs_FS');
c.refzd = fullfile(root, 'reference', '03_Outputs_ZD');
c.res   = fullfile(root, p.results);

c.seed_split = 20261020; c.test_frac = 0.20; c.val_frac = 0.20;
c.seed_frame = 20261012; c.seed_evo = 20261100; c.seed_spec = 20261204;
c.seed_airs  = 20261010; c.seed_mut = 20261010 + 7000;

c.levels = [0 1 2 4 8]; c.pop = 10; c.gen = 30; c.gen_spec = 20;
c.beta = 1; c.rho = 3; c.pm0 = 0.2; c.dnew = 2;
c.klist = [74 50 40 30 20 15 10]; c.tol = 0.005;
c.nr_att = 6000; c.nr_ben = 10000; c.nq_att = 3000; c.nq_ben = 5000; c.qblock = 1000;

c.airs = struct('ATS', 0.02, 'STIM_THRESH', 0.97, 'CLONE_RATE', 10, 'MUTATE_RATE', 2.0, ...
    'TOTAL_RES', 150, 'SEED_CELLS', 1, 'MAX_ARB_ITER', 1000);
c.npairs = 1e6;

c.rec = 0.95; c.spec = 0.8229; c.alpha = 0.005;
c.k = 3; c.trig = ["Exploitation" "WebBased" "BufferOverflow" "Bruteforce" "Benign"]; c.spec_cls = "Exploitation";
c.nb_ref = 436332; c.na_ref = 597862;
c.twins = ["DoS" "Slowloris"];
c.block = 1e7; c.refchunk = 20000;
c.workers = 2; c.threads = 2;
end
