function weights = rereferenceMatrix(isGood, method, positions, neighborCount)
%REREFERENCEMATRIX Linear re-referencing as a matrix: referenced = data * weights.
%   weights = ei.stream.rereferenceMatrix(isGood, method)
%   weights = ei.stream.rereferenceMatrix(isGood, "laplacian", positions, neighborCount)
%
%   isGood     [1 x channels] logical; bad channels never enter the reference
%              and their output column is zero (the pipeline marks them NaN)
%   method     "average" (common average of good channels), "laplacian"
%              (channel minus mean of its nearest good neighbours), or "none"
%   positions  [channels x 3] electrode coordinates, needed for "laplacian"

    arguments
        isGood (1,:) logical
        method (1,1) string {mustBeMember(method, ["average", "laplacian", "none"])}
        positions (:,3) double = zeros(0, 3)
        neighborCount (1,1) double {mustBePositive, mustBeInteger} = 4
    end

    channelCount = numel(isGood);
    goodIndex = find(isGood);
    if isempty(goodIndex)
        error("ei:stream:noGoodChannels", "All channels are marked bad; at least one good channel is required.");
    end

    switch method
        case "average"
            weights = eye(channelCount) - (double(isGood(:))/numel(goodIndex))*ones(1, channelCount);
        case "laplacian"
            if size(positions, 1) ~= channelCount
                error("ei:stream:positions", "Laplacian needs one [x y z] position per channel.");
            end
            weights = eye(channelCount);
            for j = goodIndex
                distances = vecnorm(positions(goodIndex, :) - positions(j, :), 2, 2);
                distances(goodIndex == j) = inf;
                [~, order] = sort(distances);
                neighbors = goodIndex(order(1:min(neighborCount, numel(goodIndex) - 1)));
                weights(neighbors, j) = -1/numel(neighbors);
            end
        case "none"
            weights = eye(channelCount);
        otherwise
            error("ei:stream:method", "Unknown reference method %s.", method);
    end

    weights(~isGood, :) = 0;
    weights(:, ~isGood) = 0;
end
