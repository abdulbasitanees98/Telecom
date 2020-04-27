% SNR in db to N0
function N0 = snr2n0(SNR_db)
    SNR = 10.^(SNR_db./10);
    N0  = 1 ./ SNR;
end