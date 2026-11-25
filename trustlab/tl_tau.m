function t = tl_tau(s, pos, c)
[ss, o] = sort(double(s), 'descend');
yp = pos(o); tp = cumsum(yp); fp = cumsum(~yp); nP = sum(pos); nN = sum(~pos);
eg = [ss(1:end-1) > ss(2:end); true];
iS = find(eg & fp / nN >= 1 - c.spec, 1, 'first');
f2 = 5 * tp ./ (5 * tp + 4 * (nP - tp) + fp); f2(~eg) = -Inf; [~, iF] = max(f2);
iR = find(eg & tp / nP >= c.rec, 1, 'first');
t = [ss(iS), ss(iF), ss(iR)];
end
