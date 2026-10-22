function colors = divergingColormap(levelCount)
%DIVERGINGCOLORMAP Blue (ERD, negative) - neutral gray - red (ERS, positive) colormap.
%   colors = ei.viz.divergingColormap(levelCount) returns [levelCount x 3].
%   The midpoint is a neutral gray so "no change" reads as no colour.

    arguments
        levelCount (1,1) double {mustBeInteger, mustBeGreaterThan(levelCount, 2)} = 256
    end

    blue = [42, 120, 214]/255;
    neutral = [240, 239, 236]/255;
    red = [227, 73, 72]/255;
    position = linspace(-1, 1, levelCount)';
    colors = zeros(levelCount, 3);
    isNegative = position <= 0;
    colors(isNegative, :) = neutral + (-position(isNegative)).*(blue - neutral);
    colors(~isNegative, :) = neutral + position(~isNegative).*(red - neutral);
end
