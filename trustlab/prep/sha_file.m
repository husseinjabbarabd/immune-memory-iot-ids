function h = sha_file(f)
md = java.security.MessageDigest.getInstance('SHA-256');
fid = fopen(f, 'r'); assert(fid > 0, 'Cannot open %s', f);
cl = onCleanup(@() fclose(fid));
while true
    b = fread(fid, 64 * 1024 * 1024, '*uint8');
    if isempty(b); break; end
    md.update(typecast(b, 'int8'));
end
h = lower(reshape(dec2hex(typecast(int8(md.digest()), 'uint8'), 2)', 1, []));
end
