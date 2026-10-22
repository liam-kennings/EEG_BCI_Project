function stream = replayRecording(recording, config, options)
%REPLAYRECORDING Feed a recorded file through the streaming pipeline in chunks.
%   stream = ei.pipeline.replayRecording(recording, config, Name=Value)
%   recording  struct from ei.io.readEdfRecording (Data [samples x channels] in uV)
%   config     struct from ei.config.loadConfig
%   Name=Value:
%     IsGood     channels to keep (default all)
%     Positions  [channels x 3], for Laplacian reference
%
%   stream is the 4 Hz feature stream: the fields returned by
%   StreamingPipeline.step, concatenated, plus metadata and step timing.
%   Because replay uses the online code path, these features are exactly
%   what a live run on the same data would produce.

    arguments
        recording (1,1) struct
        config (1,1) struct
        options.IsGood (1,:) logical = true(1, numel(recording.Labels))
        options.Positions (:,3) double = zeros(0, 3)
    end

    pipeline = ei.pipeline.StreamingPipeline(config, recording.Fs, recording.Labels, ...
        IsGood=options.IsGood, Positions=options.Positions);
    chunkLength = max(1, round(config.chunk_seconds*recording.Fs));
    sampleCount = size(recording.Data, 1);
    chunkCount = ceil(sampleCount/chunkLength);
    pieces = cell(chunkCount, 1);
    stepSeconds = zeros(chunkCount, 1);
    for k = 1:chunkCount
        first = (k - 1)*chunkLength + 1;
        last = min(k*chunkLength, sampleCount);
        timer = tic();
        pieces{k} = pipeline.step(recording.Data(first:last, :));
        stepSeconds(k) = toc(timer);
    end

    stream = concatenateRows([{pipeline.emptyRows()}; pieces]);
    stream.Labels = recording.Labels;
    stream.BandNames = pipeline.BandNames;
    stream.ArtifactNames = ei.stream.BandPowerStream.ArtifactNames;
    stream.IsGood = options.IsGood;
    stream.Fs = recording.Fs;
    stream.FeatureRateHz = 1/config.band_power.block_seconds;
    stream.ChunkSeconds = chunkLength/recording.Fs;
    stream.StepTimeMs = struct("median", 1000*median(stepSeconds), ...
        "p99", 1000*prctile(stepSeconds, 99), "max", 1000*max(stepSeconds));
    stream.Source = recording.SourcePath;
    stream.ConfigSha256 = config.sha256;
end

function stream = concatenateRows(pieces)
    names = fieldnames(pieces{1});
    stream = struct();
    for k = 1:numel(names)
        values = cellfun(@(piece) piece.(names{k}), pieces, UniformOutput=false);
        stream.(names{k}) = cat(1, values{:});
    end
end
