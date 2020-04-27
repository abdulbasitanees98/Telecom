function trellisDict = trellisDictionary(L, g1, g2, g3)
    % This function creates state transitions for a specific constraint
    % length and also the outputs corresponding to each input at each state
    % transition
    nStates = 2 ^ (L - 1);
    trellisDict.nStates = nStates;
    states = cellstr( dec2bin(0 : nStates - 1));
    states = split(states, '');
    states = str2double(states(:, 2 : L));
    states = [states; states];
    input = [zeros(nStates, 1); ones(nStates, 1)];
    block = [input states];
    newStates = block(:, 1 : end-1);
    trellisDict.new = bin2Dec(newStates);
    trellisDict.old = bin2Dec(states);
    trellisDict.inputs = input;
    stateTrans = [bin2Dec(states) bin2Dec(newStates)];
    trellisDict.stateTrans = stateTrans;
    
    output1 = sum(block .* g1, 2);
    output2 = sum(block .* g2, 2);
    output3 = sum(block .* g3, 2);
    output = [output1 output2 output3];
    output = mod(output, 2);
    outputSDD = output;
    outputSDD(outputSDD==0) = -1;
    trellisDict.outputs = output';
    trellisDict.outputsSDD = outputSDD';
end