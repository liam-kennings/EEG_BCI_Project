function epochs = epochFeatures(stream, events, eventTypes, options)
%EPOCHFEATURES Cut epochs from the 4 Hz feature stream around events.
%   epochs = ei.analysis.epochFeatures(stream, events, eventTypes, Name=Value)
%   stream      feature stream from ei.pipeline.replayRecording
%   events      table with OnsetSeconds, Type, Run, Trial
%   eventTypes  event types to keep, e.g. ["cue_left", "cue_right"]
%   Name=Value:
%     TminSeconds, TmaxSeconds  epoch limits relative to the event (default -2, 4)
%     Field                     "LogPower" (default) or "LogPowerDrift"
%
%   Each feature row is time-stamped at the END of its causal window, so a
%   feature at +1.0 s summarizes roughly +0.25 to +1.0 s. Epochs that leave
%   the stream or touch the filter warm-up are dropped.
%   Returns Data [trials x time x channels x bands], Time, and per-trial
%   Type, Run, Trial, OnsetSeconds.

    arguments
        stream (1,1) struct
        events table
        eventTypes (1,:) string
        options.TminSeconds (1,1) double = -2
        options.TmaxSeconds (1,1) double = 4
        options.Field (1,1) string {mustBeMember(options.Field, ["LogPower", "LogPowerDrift"])} = "LogPower"
    end

    step = 1/stream.FeatureRateHz;
    grid = (round(options.TminSeconds/step):round(options.TmaxSeconds/step))*step;
    selected = events(ismember(events.Type, eventTypes), :);
    values = stream.(options.Field);
    [~, channelCount, bandCount] = size(values);

    data = nan(height(selected), numel(grid), channelCount, bandCount);
    keep = false(height(selected), 1);
    for k = 1:height(selected)
        target = selected.OnsetSeconds(k) + grid;
        if target(1) < stream.Time(1) || target(end) > stream.Time(end)
            continue
        end
        index = interp1(stream.Time, 1:numel(stream.Time), target, "nearest");
        if any(abs(stream.Time(index)' - target) > step/2) || ~all(stream.IsFilterWarm(index))
            continue
        end
        data(k, :, :, :) = values(index, :, :);
        keep(k) = true;
    end

    epochs = struct("Data", data(keep, :, :, :), "Time", grid, "Field", options.Field, ...
        "Type", selected.Type(keep), "Run", selected.Run(keep), "Trial", selected.Trial(keep), ...
        "OnsetSeconds", selected.OnsetSeconds(keep), "Labels", stream.Labels, "BandNames", stream.BandNames);
end
