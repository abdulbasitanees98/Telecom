%% Load Audio
[music, ~] = audioread('sound.mp3', [200000, 1250000]);
yy = music(:,1);
plot(yy);
xlabel('t');
ylabel('Amplitude');

%% Histogram and pdf
voice = yy(abs(yy)~= 0);
voice = lowpass(voice, 4000, 44100);
voice = normalize(voice,'range', [-1, 1]);
plot(voice);
xlabel('t');
ylabel('Amplitude');
figure()
histo = histogram(voice, 'normalization', 'probability');
xlabel('Bins')
ylabel('Frequency')
hist_values = histo.Values;
%normal pdf
% p = fitdist(hist_values', 'Normal');
% ax = [histo.BinLimits(1):histo.BinWidth:histo.BinLimits(2)];
% pdf = normpdf(ax,p.mu,p.sigma);
% figure()
% plot(ax,pdf)
%polyfit
ax = [histo.BinLimits(1):histo.BinWidth:histo.BinLimits(2)];
degree = 10;
coeff = polyfit(ax, hist_values, degree);
y1 = polyval(coeff,ax);
figure()
plot(ax,y1)
xlabel('x');
ylabel('value');
%% Normalize pdf
fcn = @(x)(polyval(coeff,x).*((polyval(coeff,x))>0));
integr = integral(fcn, -1, 1);
coeff = coeff./integr;
y1 = polyval(coeff,ax);
y1 = y1.*(y1>0);
figure()
plot(ax,y1)
xlabel('x');
ylabel('Probability');
title('Normalized positive pdf')
%% Uniform Quantizer
N = [16, 64, 256];
D = zeros(length(N),1);
D_data_u = zeros(length(N),1);
SQNR = zeros(length(N),1);
SQNR_data_u = zeros(length(N),1);
for j = 1:length(N)
    [ai, xi, D(j), SQNR(j)] = UniformQuantizerMetrics(N(j), coeff, 'squared');
    [~, D_data_u(j), SQNR_data_u(j)] = applyQuantizer(voice, ai, xi, 'squared');
    UniformQuantizersSq(j).N  = N(j);
    UniformQuantizersSq(j).ai = ai;
    UniformQuantizersSq(j).xi = xi;
    UniformQuantizersSq(j).D_theoretical = D(j);
    UniformQuantizersSq(j).D_data = D_data_u(j);
    UniformQuantizersSq(j).SQNR_theoretical = SQNR(j);
    UniformQuantizersSq(j).SQNR_data = SQNR_data_u(j);
end
% figure()
% plot(N,D);
% xlabel('N');
% ylabel('Distortion');
% title('Uniform Quantizer - mse');
% figure()
% plot(N, SQNR);
% xlabel('N');
% ylabel('SQNR');
% title('Uniform Quantizer - mse');
%% Non uniform quantizer
N = [16, 64, 256];
D = zeros(length(N),1);
D_data_n = zeros(length(N),1);
SQNR = zeros(length(N),1);
SQNR_data_n = zeros(length(N),1);
for j = 1:length(N)
    [ai, xi, D(j), SQNR(j)] = NonUniformQuantizerMetrics(N(j), coeff, 'squared');
    [~, D_data_n(j), SQNR_data_n(j)] = applyQuantizer(voice, ai, xi, 'squared');
    NonUniformQuantizersSq(j).N  = N(j);
    NonUniformQuantizersSq(j).ai = ai;
    NonUniformQuantizersSq(j).xi = xi;
    NonUniformQuantizersSq(j).D_theoretical = D(j);
    NonUniformQuantizersSq(j).D_data = D_data_u(j);
    NonUniformQuantizersSq(j).SQNR_theoretical = SQNR(j);
    NonUniformQuantizersSq(j).SQNR_data = SQNR_data_u(j);
end
% figure()
% plot(N,D);
% xlabel('N');
% ylabel('Distortion');
% title('Non-Uniform Quantizer - mse');
% figure()
% plot(N, SQNR);
% xlabel('N');
% ylabel('SQNR');
% title('Non-Uniform Quantizer - mse');

%% Test quantizer
[music2, ~] = audioread('afterhours.mp3', [200000, 1250000]);
yyy = music2(:,1);
voice1 = yyy(abs(yyy)~= 0);
voice1 = lowpass(voice1, 4000, 44100);
voice1 = normalize(voice1,'range', [-1, 1]);
figure()
histo1 = histogram(voice1, 'normalization', 'probability');
xlabel('Bins')
ylabel('Frequency')
D_test_u = zeros(length(N),1);
SQNR_test_u = zeros(length(N),1);
for i = 1:length(N)
    [~, D_test_u(i), SQNR_test_u(i)] = applyQuantizer(voice1, UniformQuantizersSq(i).ai, UniformQuantizersSq(i).xi, 'squared');
    UniformQuantizersSq(i).D_test = D_test_u(i);
    UniformQuantizersSq(i).SQNR_test = SQNR_test_u(i);
end
D_test_n = zeros(length(N),1);
SQNR_test_n = zeros(length(N),1);
for i = 1:length(N)
    [~, D_test_n(i), SQNR_test_n(i)] = applyQuantizer(voice1, NonUniformQuantizersSq(i).ai, NonUniformQuantizersSq(i).xi, 'squared');
    NonUniformQuantizersSq(i).D_test = D_test_n(i);
    NonUniformQuantizersSq(i).SQNR_test = SQNR_test_n(i);
end

%% Part 2
%% Uniform Quantizer (absolute)
N = [16, 64, 256];
D = zeros(length(N),1);
D_data_u = zeros(length(N),1);
SQNR = zeros(length(N),1);
SQNR_data_u = zeros(length(N),1);
for j = 1:length(N)
    [ai, xi, D(j), SQNR(j)] = UniformQuantizerMetrics(N(j), coeff, 'abs');
    [~, D_data_u(j), SQNR_data_u(j)] = applyQuantizer(voice, ai, xi, 'abs');
    UniformQuantizersAbs(j).N  = N(j);
    UniformQuantizersAbs(j).ai = ai;
    UniformQuantizersAbs(j).xi = xi;
    UniformQuantizersAbs(j).D_theoretical = D(j);
    UniformQuantizersAbs(j).D_data = D_data_u(j);
    UniformQuantizersAbs(j).SQNR_theoretical = SQNR(j);
    UniformQuantizersAbs(j).SQNR_data = SQNR_data_u(j);
end
% figure()
% plot(N,D);
% xlabel('N');
% ylabel('Distortion');
% title('Uniform Quantizer - abs');
% figure()
% plot(N, SQNR);
% xlabel('N');
% ylabel('SQNR');
% title('Uniform Quantizer - abs');
%% Non uniform quantizer (absolute)
N = [16, 64, 256];
D = zeros(length(N),1);
D_data_n = zeros(length(N),1);
SQNR = zeros(length(N),1);
SQNR_data_n = zeros(length(N),1);
for j = 1:length(N)
    [ai, xi, D(j), SQNR(j)] = NonUniformQuantizerMetrics(N(j), coeff, 'abs');
    [~, D_data_n(j), SQNR_data_n(j)] = applyQuantizer(voice, ai, xi, 'abs');
    NonUniformQuantizersAbs(j).N  = N(j);
    NonUniformQuantizersAbs(j).ai = ai;
    NonUniformQuantizersAbs(j).xi = xi;
    NonUniformQuantizersAbs(j).D_theoretical = D(j);
    NonUniformQuantizersAbs(j).D_data = D_data_u(j);
    NonUniformQuantizersAbs(j).SQNR_theoretical = SQNR(j);
    NonUniformQuantizersAbs(j).SQNR_data = SQNR_data_u(j);
end
% figure()
% plot(N,D);
% xlabel('N');
% ylabel('Distortion');
% title('Non-Uniform Quantizer - abs');
% figure()
% plot(N, SQNR);
% xlabel('N');
% ylabel('SQNR');
% title('Non-Uniform Quantizer - abs');
%% Test quantizer (absolute)
[music2, Fs] = audioread('afterhours.mp3', [200000, 1250000]);
yyy = music2(:,1);
voice1 = yyy(abs(yyy)~= 0);
voice1 = lowpass(voice1, 4000, 44100);
voice1 = normalize(voice1,'range', [-1, 1]);
D_test_u = zeros(length(N),1);
SQNR_test_u = zeros(length(N),1);
for i = 1:length(N)
    [~, D_test_u(i), SQNR_test_u(i)] = applyQuantizer(voice1, UniformQuantizersAbs(i).ai, UniformQuantizersAbs(i).xi, 'abs');
    UniformQuantizersAbs(i).D_test = D_test_u(i);
    UniformQuantizersAbs(i).SQNR_test = SQNR_test_u(i);
end
D_test_n = zeros(length(N),1);
SQNR_test_n = zeros(length(N),1);
for i = 1:length(N)
    [~, D_test_n(i), SQNR_test_n(i)] = applyQuantizer(voice1, NonUniformQuantizersAbs(i).ai, NonUniformQuantizersAbs(i).xi, 'abs');
    NonUniformQuantizersAbs(i).D_test = D_test_n(i);
    NonUniformQuantizersAbs(i).SQNR_test = SQNR_test_n(i);
end

%% Comparison of non uniform quantizers using mean absolute error
for i = 1:length(N)
    NonUniformComparison(i).N = N(i);
    NonUniformComparison(i).D_abs2 = NonUniformQuantizersAbs(i).D_theoretical;
    NonUniformComparison(i).SQNR_abs2 = NonUniformQuantizersAbs(i).SQNR_theoretical;
%     NonUniformComparison(i).D_abs2_test = NonUniformQuantizersAbs(i).D_test;
%     NonUniformComparison(i).SQNR_abs2_test = NonUniformQuantizersAbs(i).SQNR_test;
    [D, SQNR] = Quantizer_Theoretical(NonUniformQuantizersSq(i).xi, NonUniformQuantizersSq(i).ai, coeff, 'abs');
    NonUniformComparison(i).D_abs1 = D;
    NonUniformComparison(i).SQNR_abs1 = SQNR;
%     [~, D, SQNR] = applyQuantizer(voice1, NonUniformQuantizersSq(i).ai, NonUniformQuantizersSq(i).xi, 'abs');
%     NonUniformComparison(i).D_abs1_test = D;
%     NonUniformComparison(i).SQNR_abs1_test = SQNR;
end

%% Part 3

voice_2 = reshape(voice, 2, length(voice)/2);
histo2  = histogram2(voice_2(1,:),voice_2(2,:), 'normalization', 'probability');
ylabel('sample i')
xlabel('samples i+1')
