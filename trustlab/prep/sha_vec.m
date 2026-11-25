function h = sha_vec(u8)
md = java.security.MessageDigest.getInstance('SHA-256');
md.update(typecast(uint8(u8(:)), 'int8'));
h = lower(reshape(dec2hex(typecast(int8(md.digest()), 'uint8'), 2)', 1, []));
end
