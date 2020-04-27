function codeword = genRandomCodeword(nBits, L)
    bits  = randi( 2, nBits, 1) - 1;
    codeword  = [bits' zeros(1, L-1)]; % L zeros sent to bring state to 0 in the end
%     codeword  = [zeros(1, L-1) sentBits]; % Add L zeros to left to simulate shift register 0 initially
end