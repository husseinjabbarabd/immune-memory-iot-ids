function run_all(selection)
if nargin < 1; selection = 'rerun'; end
assert(ismember(selection, {'rerun', 'published'}), 'selection must be ''rerun'' or ''published''');
published = strcmp(selection, 'published');
root = fileparts(mfilename('fullpath'));
addpath(root, fullfile(root, 'prep'));
inputs(fileparts(root));

pyrun(root, fullfile('prep', 'portscan_recover.py'));
prep_manifest();
prep_audit();
prep_split();
prep_split_study();
prep_features();
prep_sup();
prep_sup_split();
prep_copy();
prep_flow_split();

run_pipeline(published);
run_nested('nested', strings(0), published);
run_portfree();

pyrun(root, fullfile('py', 'grouped_split.py'));
pyrun(root, fullfile('py', 'session_split.py'));
run_grouped();

pyrun(root, fullfile('py', 'op_metrics.py'));
pyrun(root, fullfile('py', 'study_audit.py'));
pyrun(root, fullfile('py', 'reweight.py'));
pyrun(root, fullfile('py', 'study_perclass.py'));
pyrun(root, fullfile('py', 'study_reweight.py'));
pyrun(root, fullfile('py', 'study_counts.py'));
pyrun(root, fullfile('py', 'partitions.py'));
pyrun(root, fullfile('py', 'retained_flows.py'));

nested_boot();
exploit_check();
pyrun(root, fullfile('py', 'nested_table.py'), 'nested');
pyrun(root, fullfile('py', 'nested_table.py'), 'portfree');
pyrun(root, fullfile('py', 'round_outputs.py'));
pyrun(root, fullfile('py', 'nested_tables.py'));
pyrun(root, fullfile('py', 'leak_nested.py'));
pyrun(root, fullfile('py', 'fig07.py'));
pyrun(root, fullfile('py', 'fig08.py'));
pyrun(root, fullfile('py', 'fig09.py'));

pyrun(root, fullfile('py', 'portfree_closed.py'));
pyrun(root, fullfile('py', 'portfree_nested.py'));
pyrun(root, fullfile('py', 'common_far.py'));
pyrun(root, fullfile('py', 'at_far.py'));
pyrun(root, fullfile('py', 'ties.py'));
pyrun(root, fullfile('py', 'fig7b.py'));
pyrun(root, fullfile('py', 'nu_diag.py'));

run_memory();
run_select();
run_stack();
run_open();
pyrun(root, fullfile('py', 'components.py'));
end

function inputs(top)
j = jsondecode(fileread(fullfile(top, 'paths.json')));
raw = fullfile(top, j.trustlab.raw);
cls = ["API" "Benign" "Bruteforce" "BufferOverflow" "C2Beaconing" "DDoS" "DNS" "DoS" "Evasion" ...
    "Exfiltration" "Exploitation" "MITM" "Slowloris" "TLSSSL" "WebBased"];
need = [fullfile(raw, cls + ".csv"), string(fullfile(fileparts(raw), 'PortScan.csv'))];
miss = need(~arrayfun(@isfile, need));
if ~isempty(miss)
    error('TRUSTLab data not found: %s. Get the data from https://doi.org/10.82432/10317/21203', strjoin(miss, ', '));
end
end

function pyrun(root, f, varargin)
py = getenv('PYTHON'); if isempty(py); py = 'python'; end
cmd = sprintf('"%s" "%s"', py, fullfile(root, f));
if ~isempty(varargin); cmd = [cmd ' ' strjoin(varargin, ' ')]; end
assert(system(cmd) == 0, 'failed: %s', f);
end
