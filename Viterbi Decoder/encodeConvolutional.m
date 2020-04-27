function encoded = encodeConvolutional(bits, g1, g2, g3)
    L = length(g1);
    bits = [zeros(1, L-1) bits]; % Add L zeros to left to simulate shift register 0 initially
    % parity bits
    p1 = conv(bits, g1, 'valid');
    p2 = conv(bits, g2, 'valid');
    p3 = conv(bits, g3, 'valid');
    % transmission signal
    encoded = [p1; p2; p3];
    encoded = mod(encoded, 2);
end