classdef StreamingPipeline < handle
    %STREAMINGPIPELINE Causal EEG-to-feature pipeline used both offline and online.
    %   Order per chunk: re-reference -> line-noise notch -> band-pass per
    %   band and broadband -> block band power (BandPowerStream).
    %   Every stage is causal and keeps state, so feeding a recording in
    %   chunks (replay) gives exactly what a live stream would give.
    %
    %   p = ei.pipeline.StreamingPipeline(config, fs, labels, Name=Value)
    %   rows = p.step(chunk)     chunk [samples x channels] in microvolts
    %   p.reset()
    %
    %   Name=Value:
    %     IsGood     [1 x channels] logical, channels kept (default all)
    %     Positions  [channels x 3] coordinates, needed for Laplacian reference

    properties (SetAccess = private)
        Config struct
        Fs (1,1) double
        Labels (1,:) string
        IsGood (1,:) logical
        BandNames (1,:) string
    end

    properties (Access = private)
        ReferenceWeights double
        NotchFilter
        BandFilters cell
        BroadbandFilter
        BandPower
    end

    methods
        function obj = StreamingPipeline(config, fs, labels, options)
            arguments
                config (1,1) struct
                fs (1,1) double {mustBePositive}
                labels (1,:) string
                options.IsGood (1,:) logical = true(1, numel(labels))
                options.Positions (:,3) double = zeros(0, 3)
            end
            if numel(options.IsGood) ~= numel(labels)
                error("ei:pipeline:isGood", "IsGood must have one entry per channel (%d).", numel(labels));
            end
            obj.Config = config;
            obj.Fs = fs;
            obj.Labels = labels;
            obj.IsGood = options.IsGood;
            channelCount = numel(labels);

            obj.ReferenceWeights = ei.stream.rereferenceMatrix(obj.IsGood, string(config.reference.method), ...
                options.Positions, config.reference.laplacian_neighbors);

            if config.line_noise.enabled && config.line_noise.frequency_hz + config.line_noise.half_width_hz < fs/2
                edges = config.line_noise.frequency_hz + [-1, 1]*config.line_noise.half_width_hz;
                [sections, gain] = ei.dsp.designButterworth("stop", edges, config.line_noise.order, fs);
                obj.NotchFilter = ei.dsp.SosFilter(sections, gain, channelCount);
            end

            bands = reshape(config.bands, 1, []);
            obj.BandNames = string({bands.name});
            obj.BandFilters = cell(1, numel(bands));
            for b = 1:numel(bands)
                [sections, gain] = ei.dsp.designButterworth("bandpass", [bands(b).low_hz, bands(b).high_hz], ...
                    config.filter_order, fs);
                obj.BandFilters{b} = ei.dsp.SosFilter(sections, gain, channelCount);
            end
            [sections, gain] = ei.dsp.designButterworth("bandpass", ...
                [config.broadband.low_hz, config.broadband.high_hz], config.filter_order, fs);
            obj.BroadbandFilter = ei.dsp.SosFilter(sections, gain, channelCount);

            obj.BandPower = ei.stream.BandPowerStream(fs, channelCount, numel(bands), ...
                BlockSeconds=config.band_power.block_seconds, ...
                SmoothingBlocks=config.band_power.smoothing_blocks, ...
                DriftSeconds=config.drift_compensation.window_seconds, ...
                DriftEnabled=logical(config.drift_compensation.enabled), ...
                WarmupSeconds=config.filter_warmup_seconds, ...
                LogFloor=config.band_power.log_floor, ...
                IsGood=obj.IsGood);
        end

        function rows = step(obj, chunk)
            %STEP Process one chunk [samples x channels] and return completed feature blocks.
            arguments
                obj
                chunk (:,:) double
            end
            if size(chunk, 2) ~= numel(obj.Labels)
                error("ei:pipeline:channels", "Expected %d channels but the chunk has %d.", ...
                    numel(obj.Labels), size(chunk, 2));
            end
            chunk(:, ~obj.IsGood) = 0;
            referenced = chunk*obj.ReferenceWeights;
            if ~isempty(obj.NotchFilter)
                referenced = obj.NotchFilter.step(referenced);
            end
            bandCount = numel(obj.BandFilters);
            squaredBands = zeros(size(referenced, 1), size(referenced, 2), bandCount);
            for b = 1:bandCount
                squaredBands(:, :, b) = obj.BandFilters{b}.step(referenced).^2;
            end
            broadband = obj.BroadbandFilter.step(referenced);
            rows = obj.BandPower.step(squaredBands, broadband);
            rows.LogPower(:, ~obj.IsGood, :) = NaN;
            rows.LogPowerDrift(:, ~obj.IsGood, :) = NaN;
            rows.BroadbandRms(:, ~obj.IsGood) = NaN;
        end

        function reset(obj)
            %RESET Clear every filter and buffer, as at the start of a new stream.
            if ~isempty(obj.NotchFilter)
                obj.NotchFilter.reset();
            end
            for b = 1:numel(obj.BandFilters)
                obj.BandFilters{b}.reset();
            end
            obj.BroadbandFilter.reset();
            obj.BandPower.reset();
        end

        function rows = emptyRows(obj)
            %EMPTYROWS Zero-row output struct with the right field shapes.
            rows = obj.BandPower.emptyRows(0);
        end
    end
end
