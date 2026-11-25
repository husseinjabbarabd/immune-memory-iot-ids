function d = tl_data(c)
S = load(c.s09, 'classId', 'X', 'featNames'); d.y = double(S.classId(:)); d.X = S.X; d.fn = string(S.featNames(:)); clear S
E = load(c.sup, 'split'); d.sp = tl_split(d.y, double(E.split(:)), c); clear E
N = load(c.names, 'classNames', 'benignIdx'); d.names = string(N.classNames(:)); d.ben = N.benignIdx;
assert(strcmp(tl_sha(d.sp), 'c91ec90b933f612aa3112f3e9082e405ff01d25a71b31b8ea2efa2553923f85f'));
end
