function [ai, xi, D, SQNR] = UniformQuantizerMetrics(N, polynomialPdfCoeff, errorType)
    coeff = polynomialPdfCoeff;
    a1 = -1;
    an1 = 1;
    ai = a1 + (2/N)*([1:N+1]-an1);
    h = [0.5, 0.5];
    xi = conv(ai, h, 'same'); % moving average filter
    xi = xi(1:N);
    [D, SQNR] = Quantizer_Theoretical(xi, ai, coeff, errorType);
end