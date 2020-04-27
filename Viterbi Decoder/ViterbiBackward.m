function decodedBits = ViterbiBackward(viterbi,trellisDict, dataLength)
    % Traverse along the path with lowest metric
    decodedBits = zeros(dataLength,1);
    for i = dataLength+1:-1:2
        [~, newStateIdx] = min(viterbi.PM(:,i));
        newState = newStateIdx -1;
        oldState = viterbi.prevState(newStateIdx, i);
        trans = trellisDict.stateTrans == [oldState newState];
        trans = trans(:, 1) .* trans(:, 2);
        decodedBit = trellisDict.inputs(trans==1);
        decodedBits(i-1) = decodedBit;
    end
end