% The code is dynamic for any constraint length L which depends on the length
% of generator vectors. Currently implemented for k = 1
%% Non-Catastrophic code

% parity bit vectors
g1 = [1 1 0 1]; % 13
g2 = [1 0 0 1]; % 9
g3 = [1 1 1 1]; % 15
L  = length( g1); 

% Create state array and outputs for all possible inputs
trellisDict = trellisDictionary(L, g1, g2, g3);
% trellisDict.outputs(:, 2^L) % Output of Self loop

%% Implement Viterbi decoder assuming perfect receiver and channel

% nBits = 1000;
% codeword = genRandomCodeword(nBits, L);
% tx = encodeConvolutional(codeword, g1, g2, g3);
% rx = tx;
% decodedBits = ViterbiDecoder(rx, trellisDict, L, 'HDD');
% bitErrorRate = 100 * sum(codeword' ~= decodedBits)/length(decodedBits);

%% Simulate a binary symmetric channel
trials = 50;
nBits  = 1000;
p = [0, 0.001, 0.002, 0.005, 0.008 0.01:0.01:0.1, 0.15:0.05:0.5];
bitErrorRate = zeros(length(p), 1);
for i = 1:length(p)
    disp(i)
    error = zeros(trials,1);
    for j = 1:trials
        codeword = genRandomCodeword(nBits, L);
        tx = encodeConvolutional(codeword, g1, g2, g3);
        rx = bsc(tx, p(i));
        decodedBits = ViterbiDecoder(rx, trellisDict, L, 'HDD');
        error(j) = 100 * sum(codeword' ~= decodedBits)/length(decodedBits);
    end
    bitErrorRate(i) = mean(error);
end

ErrorRates.bscHDD = bitErrorRate;

figure()
loglog(p, ErrorRates.bscHDD);
xlabel('p')
ylabel('Bit error Rate')

%% Simulate transmission over AWGN uncoded

trials = 50;
nBits  = 1000;
SNR_db = [-10:2:-2, -1:10];
N0 = snr2n0(SNR_db);
bitErrorRate = zeros(length(N0), 1);
for i = 1:length(bitErrorRate)
    disp(i)
    error = zeros(trials,1);
    for j = 1:trials
        codeword = genRandomCodeword(nBits, L);
        tx = codeword;
        tx(tx==0) = -1;
        rx = AWGN(tx, N0(i));
        rx = rx > 0;
        decodedBits = rx;
        error(j) = 100 * sum(codeword ~= decodedBits)/length(decodedBits);
    end
    bitErrorRate(i) = mean(error);
end
ErrorRates.AWGNUncoded = bitErrorRate;
figure()
semilogy(SNR_db, ErrorRates.AWGNUncoded);
xlabel('Eb/N0')
ylabel('Bit error Rate')

%% Simulate transmission over AWGN channel with HDD
trials = 50;
nBits  = 1000;
SNR_db = [-10:2:-2, -1:10];
N0 = snr2n0(SNR_db);
bitErrorRate = zeros(length(N0), 1);
for i = 1:length(bitErrorRate)
    disp(i)
    error = zeros(trials,1);
    for j = 1:trials
        codeword = genRandomCodeword(nBits, L);
        tx = encodeConvolutional(codeword, g1, g2, g3);
        tx(tx==0) = -1;
        rx = AWGN(tx, N0(i));
        rx = rx > 0;
        decodedBits = ViterbiDecoder(rx, trellisDict, L, 'HDD');
        error(j) = 100 * sum(codeword' ~= decodedBits)/length(decodedBits);
    end
    bitErrorRate(i) = mean(error);
end
ErrorRates.AWGNHDD = bitErrorRate;

figure()
semilogy(SNR_db, ErrorRates.AWGNHDD);
xlabel('Eb/N0')
ylabel('Bit error Rate')

%% Simulate transmission over AWGN channel with SDD
trials = 50;
nBits  = 1000;
SNR_db = [-10:2:-2, -1:10];
N0 = snr2n0(SNR_db);
bitErrorRate = zeros(length(N0), 1);
for i = 1:length(bitErrorRate)
    disp(i)
    error = zeros(trials,1);
    for j = 1:trials
        codeword = genRandomCodeword(nBits, L);
        tx = encodeConvolutional(codeword, g1, g2, g3);
        tx(tx==0) = -1;
        rx = AWGN(tx, N0(i));
        decodedBits = ViterbiDecoder(rx, trellisDict, L, 'SDD');
        error(j) = 100 * sum(codeword' ~= decodedBits)/length(decodedBits);
    end
    bitErrorRate(i) = mean(error);
end
ErrorRates.AWGNSDD = bitErrorRate;

figure()
semilogy(SNR_db, ErrorRates.AWGNSDD);
xlabel('Eb/No')
ylabel('Bit error Rate')

%% Simulate transmission over AWGN channel using quantization with SDD

trials = 50;
nBits  = 1000;
N = [4 16];
SNR_db = [-10:2:-2, -1:10];
N0 = snr2n0(SNR_db);
bitErrorRate = zeros(length(N0), 2);
for k = 1:length(N)
    for i = 1:length(bitErrorRate)
        disp(i)
        error = zeros(trials,1);
        for j = 1:trials
            codeword = genRandomCodeword(nBits, L);
            tx = encodeConvolutional(codeword, g1, g2, g3);
            tx(tx==0) = -1;
            rx = AWGN(tx, N0(i));
            rx = UniformQuantizer(rx, N(k));
            decodedBits = ViterbiDecoder(rx, trellisDict, L, 'SDD');
            error(j) = 100 * sum(codeword' ~= decodedBits)/length(decodedBits);
        end
        bitErrorRate(i, k) = mean(error);
    end
end
ErrorRates.AWGNQSDD = bitErrorRate;

figure()
semilogy(SNR_db, ErrorRates.AWGNQSDD(:,1));
hold on
semilogy(SNR_db, ErrorRates.AWGNQSDD(:,2));
legend('N = 4', 'N = 16');
xlabel('Eb/No')
ylabel('Bit error Rate')
%% Catastrophic code

% parity bit vectors
g1 = [1 0 1 0]; %10
g2 = [1 1 0 0]; %12
g3 = [1 0 0 1]; %9
L  = length( g1); 

% Create state array and outputs for all possible inputs
trellisDict = trellisDictionary(L, g1, g2, g3);
% trellisDict.outputs(:,16)

%% Simulation  of catastrophic code over BSC

trials = 50;
nBits  = 1000;
p = [0, 0.001, 0.002, 0.005, 0.008 0.01:0.01:0.1, 0.15:0.05:0.5];
bitErrorRate = zeros(length(p), 1);
for i = 1:length(p)
    disp(i)
    error = zeros(trials,1);
    for j = 1:trials
        codeword = genRandomCodeword(nBits, L);
        tx = encodeConvolutional(codeword, g1, g2, g3);
        rx = bsc(tx, p(i));
        decodedBits = ViterbiDecoder(rx, trellisDict, L, 'HDD');
        error(j) = 100 * sum(codeword' ~= decodedBits)/length(decodedBits);
    end
    bitErrorRate(i) = mean(error);
end
ErrorRates.bscCatHDD = bitErrorRate;

figure()
loglog(p, ErrorRates.bscCatHDD);
hold on
loglog(p, ErrorRates.bscHDD);
legend('Catastrophic', 'Normal')
xlabel('p')
ylabel('Bit error Rate')

%% Simulation  of catastrophic code over AWGN with SDD

trials = 50;
nBits  = 1000;
SNR_db = [-10:2:-2, -1:10];
N0 = snr2n0(SNR_db);
bitErrorRate = zeros(length(N0), 1);
for i = 1:length(bitErrorRate)
    disp(i)
    error = zeros(trials,1);
    for j = 1:trials
        codeword = genRandomCodeword(nBits, L);
        tx = encodeConvolutional(codeword, g1, g2, g3);
        tx(tx==0) = -1;
        rx = AWGN(tx, N0(i));
        decodedBits = ViterbiDecoder(rx, trellisDict, L, 'SDD');
        error(j) = 100 * sum(codeword' ~= decodedBits)/length(decodedBits);
    end
    bitErrorRate(i) = mean(error);
end
ErrorRates.AWGNCatSDD = bitErrorRate;

figure()
semilogy(SNR_db, ErrorRates.AWGNCatSDD);
hold on
semilogy(SNR_db, ErrorRates.AWGNSDD);
legend('Catastrophic', 'Normal')
xlabel('Eb/No')
ylabel('Bit error Rate')
