%% Generate bits and initialize parameters
nBits   = 100000;
bits    = randi(2,nBits,1) - 1;
T       = 1e-6;
samples = 20;
Ts      = T / samples;

%% Part 1

%% Generate PAM signal
pulse   = genPulse(T, samples, 'Tri');
signal  = bits2signalPAM(bits, pulse(1:end-1), samples);
plot(signal(1:101))


%% Extra?
SNR_db = 10;
N0     = snr2n0(SNR_db);
r_signal = addNoise(signal, N0);
plot(r_signal(1:101))

% matched = conv(r_signal, pulse);
% r_bits  = matched(samples+1:samples:end);
% r_bits  = r_bits > 0;
% accuracy = 100 * sum(bits == r_bits)/nBits;

%% Matched filter
SNR_db = -29:2:10;
N0     = snr2n0(SNR_db);
for i=1:length(N0)
    r_signal = addNoise(signal, N0(i));
    matched = conv(r_signal, pulse);
    r_bits  = matched(samples+1:samples:end-1);
    r_bits  = r_bits > 0;
    Accuracy_tri(i).SNR_db = SNR_db(i);
    Accuracy_tri(i).accuracy = 100 * sum(bits == r_bits)/nBits;
    Accuracy_tri(i).error = 100 - Accuracy_tri(i).accuracy;
end
%%
error1 = [];
error1 = [error1, Accuracy_tri.error];
SNR = 10.^(SNR_db./10);
for i = 1:length(SNR)
    y(i) = q(sqrt(2*SNR(i)));
end
figure();
semilogy(SNR_db, error1)
xlabel('SNR');
ylabel('% error');
title('SNR (dB) vs error');
hold on
semilogy(SNR_db, 100.*y);
legend('Experimental', 'Theoretical')
%% Part 2: Matched filter more samples
T       = 1e-6;
samples = 100;
Ts      = T / samples;
pulse   = genPulse(T, samples, 'Tri');
signal  = bits2signalPAM(bits, pulse(1:end-1), samples);
SNR_db = -29:2:10;
N0     = snr2n0(SNR_db);
for i=1:length(N0)
    r_signal = addNoise(signal, N0(i));
    matched = conv(r_signal, pulse);
    r_bits  = matched(samples+1:samples:end-1);
    r_bits  = r_bits > 0;
    Accuracy_tri_samples(i).SNR_db = SNR_db(i);
    Accuracy_tri_samples(i).accuracy = 100 * sum(bits == r_bits)/nBits;
    Accuracy_tri_samples(i).error = 100 - Accuracy_tri_samples(i).accuracy;
end
error2 = [];
error2 = [error2, Accuracy_tri_samples.error];
figure();
semilogy(SNR_db, error2)
hold on
semilogy(SNR_db, error1)
hold off
xlabel('SNR');
ylabel('% error');
title('SNR (dB) vs error');
legend('100 samples', '20 samples')

%% Part 3 : Rectangular Impulse Response filter
T       = 1e-6;
samples = 20;
Ts      = T / samples;
rect_pulse = genPulse(T, samples, 'Rect');
signal  = bits2signalPAM(bits, rect_pulse(1:length(rect_pulse)-1), samples);
SNR_db = -29:2:10;
N0     = snr2n0(SNR_db);
for i=1:length(N0)
    r_signal = addNoise(signal, N0(i));
    matched = conv(r_signal, rect_pulse);
    r_bits  = matched(samples+1:samples:end-1);
    r_bits  = r_bits > 0;
    Accuracy_rect(i).SNR_db = SNR_db(i);
    Accuracy_rect(i).accuracy = 100 * sum(bits == r_bits)/nBits;
    Accuracy_rect(i).error = 100 - Accuracy_rect(i).accuracy;
end
error3 = [];
error3 = [error3, Accuracy_rect.error];
semilogy(SNR_db, error3)
hold on
semilogy(SNR_db, error1)
hold off
xlabel('SNR');
xlabel('SNR');
ylabel('% error');
title('SNR (dB) vs error');
legend('Rectangualar filter','Matched filter');
%% Part 4: Matched filter with offset in sampling (noise, bit, noise)
pulse   = genPulse(T, samples, 'Tri');
signal  = bits2signalPAM(bits, pulse(1:end-1), samples);
SNR_db = [-20:3:1, 3:2:10];
N0     = snr2n0(SNR_db);
offset = -2:2;
for m = 1:length(offset)
    for i=1:length(N0)
        for k = 1:nBits
            signal_crop = signal((samples*(k-1))+1:(samples*k)+1);
            signal_crop = addZeros(signal_crop, k, nBits, samples);
            r_signal = addNoise(signal_crop, N0(i));
            matched = conv(r_signal, pulse);
            if k==1
                r_bit  = matched(samples + 1 + offset(m));
            else
                r_bit  = matched((2*samples) + 1 + offset(m));
            end
            r_bit  = r_bit > 0;
            predict(k) = r_bit;
        end
        accuracy_4(i).SNR_db = SNR_db(i);
        accuracy_4(i).accuracy = 100 * sum(bits == predict')/nBits;
        accuracy_4(i).error = 100 - accuracy_4(i).accuracy;
    end
    AccuracyPart4(m).acc = accuracy_4;
end

for i=1:length(offset)
    error4 = [];
    error4 = [error4, AccuracyPart4(i).acc.error];
    semilogy(SNR_db, error4)
    hold on
end
hold off
xlabel('SNR');
ylabel('% error');
title('SNR (dB) vs error');
legend('-T/10','-T/20','0','T/20','T/10')
%% Part 5: Matched filter with offset in sampling (bit, bit, bit)

SNR_db = [-20:3:1, 3:2:10];
N0     = snr2n0(SNR_db);
offset = -2:2;
for m = 1:length(offset)
    for i=1:length(N0)
        for k = 1:nBits
            signal_crop = crop3(signal, k, nBits, samples);
            r_signal = addNoise(signal_crop, N0(i));
            matched = conv(r_signal, pulse);
            if k==1
                r_bit  = matched(samples + 1 + offset(m));
            else
                r_bit  = matched(2*samples + 1 + offset(m));
            end
            predict(k) = r_bit > 0;
        end
        accuracy_5(i).SNR_db = SNR_db(i);
        accuracy_5(i).accuracy = 100 * sum(bits == predict')/nBits;
        accuracy_5(i).error = 100 - accuracy_5(i).accuracy;
    end
    AccuracyPart5(m).acc = accuracy_5;
end

for i=1:length(offset)
    error5 = [];
    error5 = [error5, AccuracyPart5(i).acc.error];
    semilogy(SNR_db, error5)
    hold on
end
hold off
xlabel('SNR');
ylabel('% error');
title('SNR (dB) vs error');
legend('-T/10','-T/20','0','T/20','T/10')

%% Part 6: BFSK (Coherent)
f1 = 1 / T;
f2 = f1 + 1 / T;

samples = 100;
t = 0:T/samples:T;

basis1 = genWave(f1, samples, T, 0, 'cos');
basis2 = genWave(f2, samples, T, 0, 'cos');

[signal, sig1, sig2] = bits2signalsBFSK(bits, basis1, basis2, samples);

% plot(basis1)
% hold on
% plot(basis2)
% legend('basis1', 'basis2')
% plot(addNoise(signal(1:501), snr2n0(10)))

SNR_db      = -20:2:10;
N0       = snr2n0(SNR_db);
for i = 1:length(N0)
    r_signal = addNoise(signal, N0(i));
    r1 = correlate(r_signal, sig1, samples);
    r2 = correlate(r_signal, sig2, samples);
    predicted = r2 > r1;
    Accuracy_BFSK6(i).SNR_db = SNR_db(i);
    Accuracy_BFSK6(i).accuracy = 100 * sum(bits == predicted')/nBits;
    Accuracy_BFSK6(i).error = 100 - Accuracy_BFSK6(i).accuracy;
end
error6 = [];
error6 = [error6, Accuracy_BFSK6.error];
SNR = 10.^(SNR_db./10);
for i = 1:length(SNR)
    y2(i) = q(sqrt(SNR(i)));
end
figure();
semilogy(SNR_db, error6)
hold on
semilogy(SNR_db, 100*y2)
hold off
xlabel('SNR');
ylabel('% error');
title('SNR (dB) vs error');
legend('Experimental', 'Theoretical')
%% Part 7: BFSK (Non-coherent)

f1 = 1 / T;
f2 = f1 + 1 / T;
phi1 = 1.5;
phi2 = 0.5;

s1   = genWave(f1, samples, T, phi1, 'cos');
s2   = genWave(f2, samples, T, phi2, 'cos');

[signal, ~, ~] = bits2signalsBFSK(bits, s1, s2, samples);
% plot(signal(1:501))
% plot(addNoise(signal(1:501), snr2n0(10)))
% plot(s1)
% hold on
% plot(s2)

basis1 = genWave(f1, samples, T, 0, 'cos');
basis2 = genWave(f1, samples, T, 0, 'sin');
basis3 = genWave(f2, samples, T, 0, 'cos');
basis4 = genWave(f2, samples, T, 0, 'sin');

b1 = repSignal(basis1, nBits);
b2 = repSignal(basis2, nBits);
b3 = repSignal(basis3, nBits);
b4 = repSignal(basis4, nBits);

SNR_db = -20:2:10;
N0     = snr2n0(SNR_db);
for i = 1 : length(N0)
    r_signal = addNoise(signal, N0(i));
    x1 = correlate(r_signal, b1, samples);
    x2 = correlate(r_signal, b2, samples);
    x3 = correlate(r_signal, b3, samples);
    x4 = correlate(r_signal, b4, samples);
    x1 = x1.^2;
    x2 = x2.^2;
    x3 = x3.^2;
    x4 = x4.^2;
    r1 = x1 + x2;
    r2 = x3 + x4;
    predicted = r2 > r1;
    Accuracy_BFSK7(i).SNR_db = SNR_db(i);
    Accuracy_BFSK7(i).accuracy = 100 * sum(bits == predicted')/nBits;
    Accuracy_BFSK7(i).error = 100 - Accuracy_BFSK7(i).accuracy;
end
error7 = [];
error7 = [error7, Accuracy_BFSK7.error];
SNR = 10.^(SNR_db./10);
y3 = 0.5*exp(-SNR/2);
figure();
semilogy(SNR_db, error7)
hold on
semilogy(SNR_db, error6)
hold on
semilogy(SNR_db, 100*y3)
hold off
xlabel('SNR');
ylabel('% error');
title('SNR (dB) vs error');
legend('Non-coherent', 'Coherent', 'Theoretical (Non-coherent)')

%% Functions

% Generate normalized triangular pulse
function pulse = genPulse(T, samples, type)
    if strcmp(type, 'Tri')
        pulse  = triangularPulse(0,T,[0:T/samples:T]);
    elseif strcmp(type, 'Rect')
        pulse  = ones(1, samples+1);
    end
    energy = sqrt(sum(pulse.^2));
    pulse  = pulse ./ energy;
end

function wave = genWave(f, samples, T, phi, type)
    t      = 0:T/samples:T;
    if strcmp(type, 'cos')
        wave   = cos(2*pi*f.*t + phi);
    elseif strcmp(type, 'sin')
        wave   = -sin(2*pi*f.*t + phi);
    end
    energy = sqrt(sum(wave.^2));
    wave   = wave ./ energy;
end

% Generate corresponsing signal from bits
function signal = bits2signalPAM(bits, pulse, samplesPerBit)
    mask = repmat(bits',[samplesPerBit,1]);
    mask = mask(:);
    mask(mask == 0) = -1;
    signal = repmat(pulse', length(bits), 1);
    signal = signal .* mask;
    signal = [signal; 0];
end

% SNR in db to N0
function N0 = snr2n0(SNR_db)
    SNR = 10.^(SNR_db./10);
    N0  = 1 ./ SNR;
end

% Add noise to signal using signal and N0
function r_signal = addNoise(signal, N0)
    std      = sqrt(N0 / 2);
    noise    = std .* randn(length(signal), 1);
    r_signal = signal + noise;
end

% Crop 3 bits
function signal_crop = crop3(signal, k, nBits, samples)
    if k~=1 && k ~= nBits
        signal_crop = signal((samples*(k-2))+1:(samples*(k+1))+1);
    else
        if k==1
            signal_crop = signal(1:(2*samples) + 1);
        else
            signal_crop = signal((samples*(k-2))+1:end);
        end
    end
end

% Add zeros
function out = addZeros(signal_crop, k, nBits, samples)
    if k~=1 && k ~= nBits
        signal_crop = [zeros(samples,1); signal_crop; zeros(samples,1)];
    else
        if k==1
            signal_crop = [signal_crop; zeros(samples,1)];
        else
            signal_crop = [zeros(samples,1); signal_crop];
        end
    end
    out = signal_crop;
end

% Correlate signal with a basis (concatenated)
function r = correlate(signal, sig, samplesPerBit)
    rr    = signal .* sig;
    peak1 = rr(1:samplesPerBit:end-1);
    r11   = reshape((rr(2:end)), samplesPerBit, length(rr(2:end))/samplesPerBit);
    r111  = [peak1'; r11];
    r     = sum(r111, 1);
end

% Generate corresponsing signals from bits and basis
function [signal, sig1, sig2] = bits2signalsBFSK(bits, basis1, basis2, samplesPerBit)
    sig1 = repmat((basis1(1:end-1))', length(bits), 1);
    sig2 = repmat((basis2(1:end-1))', length(bits), 1);
    mask = repmat(bits',[samplesPerBit, 1]);
    mask = mask(:);
    signal = sig1.*(mask==0) + sig2.*(mask==1);
    signal = [signal; 1];
    sig1 = [sig1; 1];
    sig2 = [sig2; 1];
end

% Repeat signal
function sig = repSignal(basis1, nBits)
    sig = repmat((basis1(1:end-1))', nBits, 1);
    sig = [sig; 1];
end

function answer = q(x)
    syms y;
    answer=(1/sqrt(2*pi))*int(exp(-y^2/2),y,x,inf);
    answer=eval(answer);
end
