%% Part 1
%% 1: Generate bits and initialize parameters
% close all;
% clear all;
rand('seed', 774763);
M       = 100;
bits    = randi(2, M, 1);
bits(bits == 2) = -1;
T       = 1e-6;
nSamples = 200;
Ts      = T / nSamples;
pulse   = genPulse(nSamples);
signal  = bits2signalPAM(bits, pulse);
figure();
plot((0:Ts:(10*T)-Ts), signal(1:10*nSamples))
xlabel('t')
ylabel('PAM signal')

%% 1 and 2
% Add zeros (0.25T) to left and right to see effect of tau.
maxAbsShift = round(nSamples/4);
taus = [-maxAbsShift*Ts:Ts:(maxAbsShift*Ts)+Ts];
extraZeros = zeros(maxAbsShift,1);
signalNew = [extraZeros; signal; extraZeros];
% Repreat experiment 1000 times
N0 = [(0.01:0.01:0.04),(0.05:0.05:0.45),(0.5:0.5:12)];
mse = zeros(length(N0),1);
for k = 1:length(N0)
    errors = zeros(1000,1);
    for j = 1:1000
        tau = ((rand * 0.5) - 0.25)*T; % generate tau in [-0.25T, 0.25T]
        nShift = round(tau/Ts); % calculate shift in samples
        tau = taus(maxAbsShift + nShift + 1);
        if nShift > 0
            shiftedSignal = [signalNew(abs(nShift):end); zeros(abs(nShift)-1,1)];
        else
            shiftedSignal = [zeros(abs(nShift),1); signalNew(1:end-abs(nShift))];
        end
        N00 = N0(k);
        rSignal = addNoise(shiftedSignal, N00);

        %%% Matched filter
        ySignal = conv(rSignal, pulse);
        % sampled = sample(ySignal, -25, nSamples, M);
        minimize = zeros((2*maxAbsShift)+1, 1);
        for i = -maxAbsShift:maxAbsShift
            sampled = sample(ySignal, i, nSamples, M);
            minimize(maxAbsShift+1+i) = bits' * sampled;
        end
        [~, I] = max(minimize);
        shift_pr = I - maxAbsShift;
        if shift_pr <= 0
           shift_pr = shift_pr - 1; 
        end
        tau_pr = taus(maxAbsShift + shift_pr + 1);
        errors(j) = (tau - tau_pr).^2;
    end
    mse(k) = sum(errors)/1000;
end

loglog(N0(3:end), mse(3:end));
xlabel('N_0')
ylabel('MSE')
title('Symbol synchronization errors for different noise levels')
%% 3

T       = 1e-6;
nSamples = 200;
Ts      = T / nSamples;
pulse   = genPulse(nSamples);
Ms = [10, 50, 100, 300, 800, 2000];
mse = zeros(length(Ms),1);
for k = 1:length(Ms)
    disp(k);
    M       = Ms(k);
    bits    = randi(2, M, 1);
    bits(bits == 2) = -1;
    signal  = bits2signalPAM(bits, pulse);

    % Add zeros (0.25T) to left and right to see effect of tau.
    maxAbsShift = round(nSamples/4);
    taus = [-maxAbsShift*Ts:Ts:(maxAbsShift*Ts)+Ts];
    extraZeros = zeros(maxAbsShift,1);
    signalNew = [extraZeros; signal; extraZeros];
    % Repreat experiment 1000 times
    errors = zeros(1000,1);
    for j = 1:1000
        tau = ((rand * 0.5) - 0.25)*T; % generate tau in [-0.25T, 0.25T]
        nShift = round(tau/Ts); % calculate shift in samples
        tau = taus(maxAbsShift + nShift + 1);
        if nShift > 0
            shiftedSignal = [signalNew(abs(nShift):end); zeros(abs(nShift)-1,1)];
        else
            shiftedSignal = [zeros(abs(nShift),1); signalNew(1:end-abs(nShift))];
        end
        N00 = 2.5;
        rSignal = addNoise(shiftedSignal, N00);

        %%% Matched filter
        ySignal = conv(rSignal, pulse);
        % sampled = sample(ySignal, -25, nSamples, M);
        minimize = zeros((2*maxAbsShift)+1, 1);
        for i = -maxAbsShift:maxAbsShift
            sampled = sample(ySignal, i, nSamples, M);
            minimize(maxAbsShift+1+i) = bits' * sampled;
        end
        [~, I] = max(minimize);
        shift_pr = I - maxAbsShift;
        if shift_pr <= 0
           shift_pr = shift_pr - 1; 
        end
        tau_pr = taus(maxAbsShift + shift_pr + 1);
        errors(j) = (tau - tau_pr).^2;
    end
    mse(k) = sum(errors)/length(Ms);
end

semilogy(Ms, mse);
xlabel('M')
ylabel('MSE')
title('Symbol synchronization errors for different M values')

%% Part 2

Fc = 0.36e6;
T = 1e-3;
nSamples = 72000; % 2*Fc
Ts      = T / nSamples;
pulse   = genPulse(nSamples);
N0 = 2;
M       = 50;
%% Find mse using 500 experiments: Same N0, change noise samples and information symbols
phi = 1.2; % not used to estimate phi
t       = 0:T/nSamples:M*T;
t = (t(1:end-1)).';
cosWave = cos((2*pi*Fc.*t)+phi);
errors  = zeros(500,1);

for i = 1:500
    bits    = randi(2, M, 1);
    bits(bits == 2) = -1;
    signal  = bits2signalPAM(bits, pulse);
    s = signal .* cosWave;
    r = addNoise(s, N0);
    rL = r .* exp(1j * 2 * pi * Fc .* t);
    % perform correlation: Since pulse is 1 in T.
    yi = reshape(rL, 72000, M);
    yi = sum(yi);
    yReal = real(yi);
    yImag = imag(yi);
    sumReal = yReal * bits;
    sumImag = - yImag * bits;
    phi_pr = (atan(sumImag/sumReal));
    if sumImag >= 0
        if sumReal < 0
            phi_pr = phi_pr + pi;
        end
    else
        if sumReal < 0
            phi_pr = -pi + phi_pr;
        end
    end
    errors(i) = (phi - phi_pr).^2;
end
mse1 = sum(errors) / 500;

%% Fix N0 and change M
N0 = 2;
phi = 1.2; % not used to estimate phi
Ms  = [5, 15, 30, 50, 100];
mse = zeros(length(M),1);
for k = 1:length(Ms)
    disp(k)
    errors = zeros(250,1);
    M = Ms(k);
    t       = 0:T/nSamples:M*T;
    t = (t(1:end-1)).';
    cosWave = cos((2*pi*Fc.*t)+phi);
    for i = 1:250
        bits    = randi(2, M, 1);
        bits(bits == 2) = -1;
        signal  = bits2signalPAM(bits, pulse);
        s = signal .* cosWave;
        r = addNoise(s, N0);
        rL = r .* exp(1j * 2 * pi * Fc .* t);
        % perform correlation: Since pulse is 1 in T.
        yi = reshape(rL, 72000, M);
        yi = sum(yi);
        yReal = real(yi);
        yImag = imag(yi);
        sumReal = yReal * bits;
        sumImag = - yImag * bits;
        phi_pr = (atan(sumImag/sumReal));
        if sumImag >= 0
            if sumReal < 0
                phi_pr = phi_pr + pi;
            end
        else
            if sumReal < 0
                phi_pr = -pi + phi_pr;
            end
        end
        errors(i) = (phi - phi_pr).^2;
    end
    mse(k) = sum(errors) / 250;
end
plot(Ms, mse)
xlabel('M')
ylabel('MSE')
title('MSE for carrier phase for different values of M')


%% Fix M and change N0
phi = 1.2; % not used to estimate phi
M  = 20;
t       = 0:T/nSamples:M*T;
t = (t(1:end-1)).';
cosWave = cos((2*pi*Fc.*t)+phi);
N0s = [0.01, 0.05, 0.1, 0.5, 1, 3, 5, 10];
mse = zeros(length(N0s),1);
for k = 1:length(N0s)
    disp(k)
    errors = zeros(250,1);
    N0 = N0s(k);
    for i = 1:250
        bits    = randi(2, M, 1);
        bits(bits == 2) = -1;
        signal  = bits2signalPAM(bits, pulse);
        s = signal .* cosWave;
        r = addNoise(s, N0);
        rL = r .* exp(1j * 2 * pi * Fc .* t);
        % perform correlation: Since pulse is 1 in T.
        yi = reshape(rL, nSamples, M);
        yi = sum(yi);
        yReal = real(yi);
        yImag = imag(yi);
        sumReal = yReal * bits;
        sumImag = - yImag * bits;
        phi_pr = (atan(sumImag/sumReal));
        if sumImag >= 0
            if sumReal < 0
                phi_pr = phi_pr + pi;
            end
        else
            if sumReal < 0
                phi_pr = -pi + phi_pr;
            end
        end
        errors(i) = (phi - phi_pr).^2;
    end
    mse(k) = sum(errors) / 250;
end
plot(N0s, mse)
xlabel('N_0')
ylabel('MSE')
title('MSE for carrier phase for different values of N_0')

%% Functions

function ym = sample(signal, nShift, nSamples, M)
    maxAbsShift = round(nSamples/4); 
    ym = signal(nSamples+maxAbsShift-nShift:nSamples:end);
    ym = ym(1:M);
end

% The following functions have been taken from my EEE 431 telecom projects
% coded by me
% Generate normalized triangular pulse
function pulse = genPulse(samples)
%     stepSize = T/samples;
    pulse  = ones(samples, 1);
%     energy = sqrt(sum(pulse.^2));
%     pulse  = pulse ./ energy;
end

% Generate corresponsing signal from bits
function signal = bits2signalPAM(bits, pulse)
    samplesPerBit = length(pulse);
    mask = repmat(bits',[samplesPerBit,1]);
    mask = mask(:);
%     mask(mask == 0) = -1;
    signal = repmat(pulse, length(bits), 1);
    signal = signal .* mask;
%     signal = [signal; 0];
end

% Add noise to signal using signal and N0
function r_signal = addNoise(signal, N0)
    std      = sqrt(N0 / 2);
    noise    = std .* randn(length(signal), 1);
    r_signal = signal + noise;
end
