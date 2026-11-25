function tl_pool(c)
if maxNumCompThreads ~= c.threads; maxNumCompThreads(c.threads); end
p = gcp('nocreate');
if ~isempty(p) && p.NumWorkers ~= c.workers; delete(p); p = []; end
if isempty(p); parpool('Processes', c.workers, 'IdleTimeout', 120); end
end
