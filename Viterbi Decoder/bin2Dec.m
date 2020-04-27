function dec = bin2Dec(bin)
    n = size(bin);
    n = n(2);
    vec = 2.^[n-1:-1:0];
    dec = sum(bin.*vec, 2);
end