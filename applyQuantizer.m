function [xq, D, SQNR] = applyQuantizer(voice, ai, xi, errorType)
    xq = zeros(length(voice),1);
    for i = 1:length(xi)
        xq((voice < ai(i+1))&(voice >= ai(i))) = xi(i);
    end
    if strcmp(errorType, 'squared')
        D = mean((voice-xq).^2);
    elseif strcmp(errorType, 'abs')
        D = mean(abs(voice-xq));
    else
        error('Unknown error type')
    end
    SQNR = mean(voice.^2)/D;
end