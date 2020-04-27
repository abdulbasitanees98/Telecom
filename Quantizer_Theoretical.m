function [D, SQNR] = Quantizer_Theoretical(xi, ai, coeff, errorType)
    D = 0;
    for i = 1:length(xi)
        if strcmp(errorType, 'squared')
            func1 = @(x)((x-xi(i)).^2).*(polyval(coeff,x).*((polyval(coeff,x))>0));
        elseif strcmp(errorType, 'abs')
            func1 = @(x)(abs(x-xi(i))).*(polyval(coeff,x).*((polyval(coeff,x))>0));
        else
            error('Invalid error type')
        end
        D = D + integral(func1, ai(i), ai(i+1));
    end
    func2 = @(x)((x.^2).*(polyval(coeff,x).*((polyval(coeff,x))>0)));
    P     = integral(func2, -1, 1);
    SQNR  = P/D;
end