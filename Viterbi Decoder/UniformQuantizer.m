function rxq = UniformQuantizer(rx, N)
    % This function is used from my telecom I course project 1    
    a1 = -1;
    an1 = 1;
    ai = a1 + (2/N)*([1:N+1]-an1);
    h = [0.5, 0.5];
    xi = conv(ai, h, 'same'); % moving average filter
    xi = xi(1:N);
    rxq = zeros(size(rx));
    rxq(rx < ai(1)) = xi(1);
    for i = 1:length(xi)
        rxq((rx < ai(i+1))&(rx >= ai(i))) = xi(i);
    end
    rxq((rx >= ai(N+1))) = xi(N);
end