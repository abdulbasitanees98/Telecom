function [D, SQNR] = evaluateQuantizer(voice, ai, xi)
    xq = zeros(length(voice),1);
    for i = 1:N
        xq((voice < ai(i+1))&(voice >= ai(i))) = xi(i);
    end
    D = mean((voice-xq).^2);
    SQNR = mean(voice.^2)/D2;
end