classdef BandPowerStream < handle
    %BANDPOWERSTREAM Causal block band power with trailing smoothing and drift compensation.
    %   Implements the feature stage of the plan (after Lee et al. 2022,
    %   made fully causal):
    %     1. mean of the squared band-passed signal in each block (0.25 s)
    %     2. trailing mean over the last SmoothingBlocks blocks (0.75 s)
    %     3. log10
    %     4. drift compensation: subtract the mean log power of the last
    %        DriftBlocks blocks (25 s), current block included
    %   It also summarizes the broadband signal per block (peak-to-peak and
    %   RMS) for the artifact gate and the amplitude-only baseline.
    %
    %   Block k ends at sample floor(k*BlockSeconds*Fs), so non-integer block
    %   lengths (e.g. 62.5 samples at 250 Hz) are handled without drift.
    %
    %   s = ei.stream.BandPowerStream(fs, channelCount, bandCount, Name=Value)
    %   rows = s.step(squaredBands, broadband)
    %       squaredBands  [samples x channels x bands] squared band-passed signal
    %       broadband     [samples x channels] broadband-filtered signal
    %       rows          struct of completed blocks (see emptyRows)

    properties (SetAccess = private)
        Fs (1,1) double
        ChannelCount (1,1) double
        BandCount (1,1) double
        BlockSeconds (1,1) double
        SmoothingBlocks (1,1) double
        DriftBlocks (1,1) double
        DriftEnabled (1,1) logical
        WarmupSeconds (1,1) double
        LogFloor (1,1) double
        IsGood (1,:) logical
    end

    properties (Access = private)
        SamplesSeen (1,1) double = 0
        BlockIndex (1,1) double = 0
        BlockSampleCount (1,1) double = 0
        PowerSum double
        BroadMax double
        BroadMin double
        BroadSumSquares double
        SmoothBuffer double
        SmoothCount (1,1) double = 0
        SmoothPointer (1,1) double = 0
        DriftBuffer double
        DriftCount (1,1) double = 0
        DriftPointer (1,1) double = 0
    end

    properties (Constant)
        ArtifactNames = ["max_peak_to_peak_uv", "median_peak_to_peak_uv", "median_rms_uv", "max_rms_uv"]
    end

    methods
        function obj = BandPowerStream(fs, channelCount, bandCount, options)
            arguments
                fs (1,1) double {mustBePositive}
                channelCount (1,1) double {mustBePositive, mustBeInteger}
                bandCount (1,1) double {mustBePositive, mustBeInteger}
                options.BlockSeconds (1,1) double {mustBePositive} = 0.25
                options.SmoothingBlocks (1,1) double {mustBePositive, mustBeInteger} = 3
                options.DriftSeconds (1,1) double {mustBePositive} = 25
                options.DriftEnabled (1,1) logical = true
                options.WarmupSeconds (1,1) double {mustBeNonnegative} = 2
                options.LogFloor (1,1) double {mustBePositive} = 1e-12
                options.IsGood (1,:) logical = true(1, channelCount)
            end
            if fs*options.BlockSeconds < 1
                error("ei:stream:block", "A %.3f s block holds fewer than one sample at %.1f Hz.", ...
                    options.BlockSeconds, fs);
            end
            obj.Fs = fs;
            obj.ChannelCount = channelCount;
            obj.BandCount = bandCount;
            obj.BlockSeconds = options.BlockSeconds;
            obj.SmoothingBlocks = options.SmoothingBlocks;
            obj.DriftBlocks = max(1, round(options.DriftSeconds/options.BlockSeconds));
            obj.DriftEnabled = options.DriftEnabled;
            obj.WarmupSeconds = options.WarmupSeconds;
            obj.LogFloor = options.LogFloor;
            obj.IsGood = options.IsGood;
            obj.reset();
        end

        function rows = step(obj, squaredBands, broadband)
            %STEP Consume one chunk and return every block it completes.
            arguments
                obj
                squaredBands (:,:,:) double
                broadband (:,:) double
            end
            sampleCount = size(squaredBands, 1);
            if size(broadband, 1) ~= sampleCount
                error("ei:stream:alignment", "squaredBands and broadband must have the same number of samples.");
            end
            maxRows = floor((obj.SamplesSeen + sampleCount)/(obj.BlockSeconds*obj.Fs)) - obj.BlockIndex + 1;
            rows = obj.emptyRows(max(maxRows, 0));
            rowCount = 0;
            position = 1;
            while position <= sampleCount
                boundary = floor((obj.BlockIndex + 1)*obj.BlockSeconds*obj.Fs);
                take = min(boundary - obj.SamplesSeen, sampleCount - position + 1);
                segment = position:(position + take - 1);
                obj.accumulate(squaredBands(segment, :, :), broadband(segment, :));
                obj.SamplesSeen = obj.SamplesSeen + take;
                position = position + take;
                if obj.SamplesSeen == boundary
                    rowCount = rowCount + 1;
                    rows = obj.finalizeBlock(rows, rowCount);
                end
            end
            rows = ei.stream.BandPowerStream.trimRows(rows, rowCount);
        end

        function reset(obj)
            %RESET Discard all buffered samples and history.
            obj.SamplesSeen = 0;
            obj.BlockIndex = 0;
            obj.SmoothBuffer = zeros(obj.SmoothingBlocks, obj.ChannelCount, obj.BandCount);
            obj.SmoothCount = 0;
            obj.SmoothPointer = 0;
            obj.DriftBuffer = zeros(obj.DriftBlocks, obj.ChannelCount, obj.BandCount);
            obj.DriftCount = 0;
            obj.DriftPointer = 0;
            obj.clearBlock();
        end

        function rows = emptyRows(obj, rowCount)
            %EMPTYROWS Preallocated output struct for rowCount blocks.
            rows = struct( ...
                "Time", zeros(rowCount, 1), ...
                "LogPower", nan(rowCount, obj.ChannelCount, obj.BandCount), ...
                "LogPowerDrift", nan(rowCount, obj.ChannelCount, obj.BandCount), ...
                "BroadbandRms", nan(rowCount, obj.ChannelCount), ...
                "Artifact", nan(rowCount, numel(obj.ArtifactNames)), ...
                "IsFilterWarm", false(rowCount, 1), ...
                "IsDriftWarm", false(rowCount, 1));
        end
    end

    methods (Access = private)
        function accumulate(obj, squaredBands, broadband)
            obj.PowerSum = obj.PowerSum + reshape(sum(squaredBands, 1), obj.ChannelCount, obj.BandCount);
            obj.BroadMax = max(obj.BroadMax, max(broadband, [], 1));
            obj.BroadMin = min(obj.BroadMin, min(broadband, [], 1));
            obj.BroadSumSquares = obj.BroadSumSquares + sum(broadband.^2, 1);
            obj.BlockSampleCount = obj.BlockSampleCount + size(squaredBands, 1);
        end

        function rows = finalizeBlock(obj, rows, rowIndex)
            power = obj.PowerSum/obj.BlockSampleCount;

            obj.SmoothPointer = mod(obj.SmoothPointer, obj.SmoothingBlocks) + 1;
            obj.SmoothBuffer(obj.SmoothPointer, :, :) = reshape(power, 1, obj.ChannelCount, obj.BandCount);
            obj.SmoothCount = min(obj.SmoothCount + 1, obj.SmoothingBlocks);
            smoothed = reshape(mean(obj.SmoothBuffer(1:obj.SmoothCount, :, :), 1), obj.ChannelCount, obj.BandCount);
            logPower = log10(max(smoothed, obj.LogFloor));

            obj.DriftPointer = mod(obj.DriftPointer, obj.DriftBlocks) + 1;
            obj.DriftBuffer(obj.DriftPointer, :, :) = reshape(logPower, 1, obj.ChannelCount, obj.BandCount);
            obj.DriftCount = min(obj.DriftCount + 1, obj.DriftBlocks);
            if obj.DriftEnabled
                driftMean = reshape(mean(obj.DriftBuffer(1:obj.DriftCount, :, :), 1), obj.ChannelCount, obj.BandCount);
            else
                driftMean = zeros(obj.ChannelCount, obj.BandCount);
            end

            peakToPeak = obj.BroadMax - obj.BroadMin;
            rms = sqrt(obj.BroadSumSquares/obj.BlockSampleCount);
            good = obj.IsGood;

            obj.BlockIndex = obj.BlockIndex + 1;
            time = obj.SamplesSeen/obj.Fs;
            rows.Time(rowIndex) = time;
            rows.LogPower(rowIndex, :, :) = reshape(logPower, 1, obj.ChannelCount, obj.BandCount);
            rows.LogPowerDrift(rowIndex, :, :) = reshape(logPower - driftMean, 1, obj.ChannelCount, obj.BandCount);
            rows.BroadbandRms(rowIndex, :) = rms;
            rows.Artifact(rowIndex, :) = [max(peakToPeak(good)), median(peakToPeak(good)), ...
                median(rms(good)), max(rms(good))];
            rows.IsFilterWarm(rowIndex) = time >= obj.WarmupSeconds;
            rows.IsDriftWarm(rowIndex) = obj.DriftCount >= obj.DriftBlocks;
            obj.clearBlock();
        end

        function clearBlock(obj)
            obj.BlockSampleCount = 0;
            obj.PowerSum = zeros(obj.ChannelCount, obj.BandCount);
            obj.BroadMax = -inf(1, obj.ChannelCount);
            obj.BroadMin = inf(1, obj.ChannelCount);
            obj.BroadSumSquares = zeros(1, obj.ChannelCount);
        end
    end

    methods (Static, Access = private)
        function rows = trimRows(rows, rowCount)
            names = fieldnames(rows);
            for k = 1:numel(names)
                values = rows.(names{k});
                index = repmat({':'}, 1, ndims(values));
                index{1} = 1:rowCount;
                rows.(names{k}) = values(index{:});
            end
        end
    end
end
