function [ai, xi, D, SQNR] = NonUniformQuantizerMetrics(N, polynomialPdfCoeff, errorType)
    coeff = polynomialPdfCoeff;
    a1 = -1;
    an1 = 1;
    ai = a1 + (2/N)*([1:N+1]-an1);
    if strcmp(errorType, 'squared')
        num = @(x)((x).*(polyval(coeff,x).*((polyval(coeff,x))>0)));
        den = @(x)((polyval(coeff,x).*((polyval(coeff,x))>0)));
    end
    for k = 1:50
        if strcmp(errorType, 'squared')
            for i = 1:N
               xi(i) = integral(num, ai(i), ai(i+1))/integral(den, ai(i), ai(i+1));
               if isnan(xi(i))
                  xi(i) = (ai(i)+ ai(i+1))/2;
               end
            end
        elseif strcmp(errorType, 'abs')
            nSpacing = 20; 
            for i = 1:N
               xi_i = linspace(ai(i),ai(i+1), nSpacing);
               for m = 1:nSpacing
                   func = @(x)(abs(x-xi_i(m)).*(polyval(coeff,x).*((polyval(coeff,x))>0)));
                   integr(m) = integral(func, ai(i), ai(i+1));
               end
               [~, ind] = min(integr);
               xi(i) = xi_i(ind);
            end
        else
            error('Unknown error type')
        end
        h = [0.5, 0.5];
        ai = conv(h, xi);
        ai(1) = -1;
        ai(N+1) = 1;
    end
    [D, SQNR] = Quantizer_Theoretical(xi, ai, coeff, errorType);
end