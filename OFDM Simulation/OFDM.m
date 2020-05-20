%% OFDM simulation

f00 = 875e6;
fn = 882.68e6;
B  = 7.68e6;
dB = 15e3;
f0 = 875e6 + (dB/2);
T  = 1 / B;

N  = 512;
Ms = [4, 16, 64];
snr = 4:2:20;
N0s = snr2n0(snr);
nSymbols = 10;
iters = 1;
BER = zeros(3, length(N0s));
totalBits = nSymbols*6*N;
for j = 1:length(snr)
    disp(j)
    errors = zeros(1,iters);
    for m = 1:iters
        bits = randi(2, totalBits, 1) - 1; % assuming that the bits are sampled at k*B
        for i = 1:length(Ms)
            M  = Ms(i);
            k = log2(Ms(i));
            bitsP = reshape(bits, length(bits) / k, k); % which makes the sampling frequency of numbers as B
%             X = bin2Dec(bitsP);
              X = bi2de(bitsP);
            X = reshape(X, nSymbols *6/k, N);
            Xmod = qammod(X, M, 'bin');
            tx = ifft(Xmod, N, 2);
            rx = addNoise(tx, snr(j));
%             rx = awgn(tx, snr(j), 'measured');
            Y = fft(rx, N, 2);
            Yf = qamdemod(Y, M, 'bin');
            Yf2 = de2bi(Yf(:),k);
%           Yf2 = dec2Bin(Yf(:),k);
            errors(i,m) = sum(bits(:) ~= Yf2(:)) / totalBits;
        end
    end
    BER(:, j) = mean(errors,2);
end

semilogy(snr, BER(1,:))
hold on
semilogy(snr, BER(2,:))
hold on
semilogy(snr, BER(3,:))
legend('QPSK', '16 QAM', '64 QAM')
title('OFDM BER for various noise levels - Binary Coding')
xlabel('SNR_d_b')
ylabel('BER')

%% Functions

% Add noise to signal using signal and snr
function r_signal = addNoise(signal, snrDb)
    signalDb = 10 * log10(std(signal(:))^2);
    noiseDb  = signalDb - snrDb; 
    noise    = (10^(noiseDb/10/2))*randn(size(signal)); 
    r_signal = signal + noise;
end

function N0 = snr2n0(SNR_db)
    SNR = 10.^(SNR_db./10);
    N0  = 1 ./ SNR;
end

function dec = bin2Dec(bin)
    n = size(bin);
    n = n(2);
    vec = 2.^[n-1:-1:0];
    dec = sum(bin.*vec, 2);
end

function bin = dec2Bin(dec, k)
    a = dec2bin(dec(:));
    a = cellstr(a);
    a = split(a, '');
    bin = str2double(a(:,2:k+1));
end
