function rx = bsc(tx, p)
    % Simulates a binary symmetric channel
    sizetx = size(tx);
    nBits = max(sizetx);
    nParities = min(sizetx);
    nRx = nBits * nParities;
    nOnes = round(nRx * p);
    noise = [ones(1, nOnes) zeros(1, nRx - nOnes)];
    shuffledIndex = randperm(nRx);
    noise = noise(shuffledIndex);
    noise = reshape(noise, nParities, nRx/nParities);
    rx = mod((tx + noise), 2); 
end