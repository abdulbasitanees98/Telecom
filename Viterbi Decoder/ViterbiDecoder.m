function decodedBits = ViterbiDecoder(rx, trellisDict, L, type)
    nStates = trellisDict.nStates;
    dataLength = size(rx);
    stateTrans = trellisDict.stateTrans;
    
    % Initialize
    for j = 1:L
        for i = 1:nStates
            viterbi.nextState(i,j).val = []; 
        end
    end
    viterbi.PM = 1e9 .* ones(nStates, dataLength(2)+1); % Initialize as a bignumber so that it will be replaced
    viterbi.nextState(1,1).val = stateTrans(stateTrans(:,1)==0, 2); % start from state zero and propagate forward
    viterbi.PM(1,1) = 0;

    % propagate from one state to next
    for time = 1:dataLength(2)
        for state = 0:nStates-1
            % Find connection of current state to next states
            % after index L, we get next state from Lth index since it will be same everywhere afterwards
            prev2cur = viterbi.nextState(state+1, (time * (time < L)) + ( L * (time >= L))).val;
            if ~isempty(prev2cur)
                % Find corresponding output code for a specific state
                % transition and calculate path metric
                trans1 = stateTrans == [state prev2cur(1)]; trans1 = trans1(:, 1) .* trans1(:, 2);
                trans2 = stateTrans == [state prev2cur(2)]; trans2 = trans2(:, 1) .* trans2(:, 2);
                if strcmp(type, 'HDD')
                    out0 = trellisDict.outputs(:,trans1 == 1);
                    out1 = trellisDict.outputs(:,trans2 == 1);
                    PM1 = sum(rx(:, time) ~= out0);
                    PM2 = sum(rx(:, time) ~= out1);
                elseif strcmp(type, 'SDD')
                    out0 = trellisDict.outputsSDD(:,trans1 == 1);
                    out1 = trellisDict.outputsSDD(:,trans2 == 1);
                    % Calculate negative (instead of positive) correlation so that PM is minimized in consistency with HDD
                    PM1 = - out0' * rx(:, time);
                    PM2 = - out1' * rx(:, time);
                else
                    error('Enter valid decoding type: HDD or SDD')
                end
                % update path metrics and add connections if suitable path
                if viterbi.PM(prev2cur(1)+1, time+1) > viterbi.PM(state+1, time)+ PM1
                    viterbi.PM(prev2cur(1)+1, time+1) = viterbi.PM(state+1, time)+ PM1;
                    viterbi.prevState(prev2cur(1)+1, time+1) = state;
                end
                if viterbi.PM(prev2cur(2)+1, time+1) > viterbi.PM(state+1, time)+ PM2
                    viterbi.PM(prev2cur(2)+1, time+1) = viterbi.PM(state+1, time)+ PM2;
                    viterbi.prevState(prev2cur(2)+1, time+1) = state;
                end
                % find state connections
                if time < L
                    viterbi.nextState(prev2cur(1)+1, time+1).val = stateTrans(stateTrans(:,1) == prev2cur(1), 2);
                    viterbi.nextState(prev2cur(2)+1, time+1).val = stateTrans(stateTrans(:,1) == prev2cur(2), 2);
                end
            end
        end
    end
    decodedBits = ViterbiBackward(viterbi, trellisDict, dataLength(2));
end