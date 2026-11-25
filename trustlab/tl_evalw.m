function f = tl_evalw(W, K)
persistent M
V = K.Value;
id = sprintf('%s_%d', V.id, V.tar); clear V
if isempty(M) || ~strcmp(M.id, id)
    M = struct('id', id, 'map', containers.Map('KeyType', 'char', 'ValueType', 'double'));
end
key = cellstr(char(W + 48));
[ku, ~, j] = unique(key);
new = find(~isKey(M.map, ku));
if ~isempty(new)
    Wn = W(cellfun(@(x) find(strcmp(key, x), 1), ku(new)), :);
    fn = zeros(numel(new), 1);
    parfor i = 1:numel(new); fn(i) = tl_fitx(Wn(i, :), K.Value); end
    for i = 1:numel(new); M.map(ku{new(i)}) = fn(i); end
end
fu = cell2mat(values(M.map, ku));
f = fu(j);
end
