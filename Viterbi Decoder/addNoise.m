% Add noise to signal using signal and N0
function rx = addNoise(tx, N0)
    std      = sqrt(N0 / 2);
    noise    = std .* randn(size(tx));
    rx = tx + noise;
end