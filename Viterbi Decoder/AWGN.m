function rx = AWGN(tx, N0)
    % Simulates an AWGN channel
    std      = sqrt(N0 / 2);
    noise    = std .* randn(size(tx));
    rx = tx + noise;
end