function [isGoodChannel, isGoodRun] = combineRunQuality(isBadByRun, maxBadFraction)
%COMBINERUNQUALITY Reject runs with too many bad channels, then pool bad channels.
%   [isGoodChannel, isGoodRun] = ei.qc.combineRunQuality(isBadByRun, maxBadFraction)
%   isBadByRun  [runs x channels] logical, from ei.qc.findBadChannels per run
%   A run is rejected when more than maxBadFraction of its channels are bad.
%   A channel bad in any kept run is bad in all runs (Lee et al. 2022).

    arguments
        isBadByRun (:,:) logical
        maxBadFraction (1,1) double {mustBeBetween(maxBadFraction, 0, 1)}
    end

    isGoodRun = mean(isBadByRun, 2) <= maxBadFraction;
    isGoodChannel = ~any(isBadByRun(isGoodRun, :), 1);
end
