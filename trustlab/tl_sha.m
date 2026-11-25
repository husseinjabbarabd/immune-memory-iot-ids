function h = tl_sha(u8)
md = java.security.MessageDigest.getInstance('SHA-256');
u8 = uint8(u8(:)); n = 32 * 1024 * 1024;
for s = 1:n:numel(u8)
    md.update(typecast(u8(s:min(s + n - 1, end)), 'int8'));
end
h = lower(reshape(dec2hex(typecast(int8(md.digest()), 'uint8'), 2)', 1, []));
end
