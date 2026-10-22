function summary = runPhysionetDemo(options)
%RUNPHYSIONETDEMO End-to-end check on PhysioNet EEGMMIDB motor imagery (runs 4, 8, 12).
%   summary = runPhysionetDemo(Name=Value)
%   Name=Value:
%     Subjects      subject numbers (default 1:20; 88, 89, 92, 100 are always skipped
%                   because of known timing/label defects)
%     OutputFolder  default outputs/demo_physionet
%
%   For each subject: per-run channel quality (Lee et al. rule, robust z),
%   causal replay through the streaming pipeline, epoching of the 4 Hz
%   feature stream around left/right imagery cues, and ERD/S in dB against
%   the pre-cue reference. Writes:
%     quality_S###.png               channels x runs grid (first subject)
%     erds_topomaps_mu.png / _beta   grand-average ERD/S, left, right, right-left
%     erds_timecourse_mu.png / _beta C3, Cz, C4 with 95% CI across subjects
%     features_S###R04.h5            one feature stream, for the Python side
%     run_manifest.json              config hash, git commit, exclusions, counts, timing

    arguments
        options.Subjects (1,:) double {mustBeInteger, mustBePositive} = 1:20
        options.OutputFolder (1,1) string = fullfile(ei.util.projectRoot(), "outputs", "demo_physionet")
    end

    config = ei.config.loadConfig();
    dataRoot = fullfile(ei.util.projectRoot(), config.data.physionet_root);
    runs = [4, 8, 12];
    knownDefects = [88, 89, 92, 100];
    subjects = setdiff(options.Subjects, knownDefects, "stable");
    if ~isfolder(options.OutputFolder)
        mkdir(options.OutputFolder);
    end

    conditions = ["cue_left", "cue_right"];
    subjectErds = cell(numel(subjects), numel(conditions));
    records = struct("Subject", {}, "BadChannels", {}, "RejectedRuns", {}, "Trials", {});
    stepTimes = [];
    labels = strings(1, 0);
    for s = 1:numel(subjects)
        subject = subjects(s);
        recordings = arrayfun(@(run) ei.io.readEdfRecording(ei.io.findPhysionetRun(dataRoot, subject, run)), runs);
        labels = recordings(1).Labels;
        isBadByRun = false(numel(runs), numel(labels));
        for r = 1:numel(runs)
            quality = ei.qc.findBadChannels(recordings(r).Data, recordings(r).Fs, config);
            isBadByRun(r, :) = quality.IsBad;
        end
        [isGood, isGoodRun] = ei.qc.combineRunQuality(isBadByRun, config.channel_health.max_bad_fraction_per_run);
        if s == 1
            [~, positions] = ei.io.channelLocations(labels);
            figureHandle = ei.viz.plotChannelQuality(isBadByRun, labels, "R" + string(runs), IsGoodRun=isGoodRun, ...
                Title=sprintf("S%03d channel quality (red = rejected)", subject));
            exportgraphics(figureHandle, fullfile(options.OutputFolder, sprintf("quality_S%03d.png", subject)), Resolution=150);
            close(figureHandle);
        end

        epochsByRun = cell(1, numel(runs));
        for r = find(isGoodRun')
            stream = ei.pipeline.replayRecording(recordings(r), config, IsGood=isGood, Positions=positions);
            stepTimes(end + 1, :) = [stream.StepTimeMs.median, stream.StepTimeMs.p99]; %#ok<AGROW>
            if s == 1 && runs(r) == 4
                stream.Source = recordings(r).SourcePath;
                ei.io.writeFeatureStream(stream, fullfile(options.OutputFolder, sprintf("features_S%03dR%02d.h5", subject, runs(r))));
            end
            events = ei.io.physionetEvents(recordings(r).Annotations, runs(r));
            epochsByRun{r} = ei.analysis.epochFeatures(stream, events, conditions, ...
                TminSeconds=config.epoch.tmin_seconds, TmaxSeconds=config.epoch.tmax_seconds);
        end
        epochs = mergeEpochs(epochsByRun(~cellfun(@isempty, epochsByRun)));
        erds = ei.analysis.computeErds(epochs, reshape(config.erds.reference_window_seconds, 1, 2));
        trialCounts = zeros(1, numel(conditions));
        for c = 1:numel(conditions)
            selected = erds.Type == conditions(c);
            trialCounts(c) = nnz(selected);
            subjectErds{s, c} = squeeze(mean(erds.Data(selected, :, :, :), 1));
        end
        records(end + 1) = struct("Subject", subject, "BadChannels", labels(~isGood), ...
            "RejectedRuns", runs(~isGoodRun), "Trials", trialCounts); %#ok<AGROW>
        fprintf("S%03d: %d bad channels, %d rejected runs, trials left/right %d/%d\n", subject, ...
            nnz(~isGood), nnz(~isGoodRun), trialCounts(1), trialCounts(2));
    end

    time = erds.Time;
    [chanlocs, ~] = ei.io.channelLocations(labels);
    bandNames = erds.BandNames;
    centers = reshape(config.topomap.time_bin_centers_seconds, 1, []);
    halfWidth = config.topomap.time_bin_width_seconds/2;
    figurePaths = strings(1, 0);
    for b = 1:numel(bandNames)
        perSubject = cellfun(@(values) values(:, :, b), subjectErds, UniformOutput=false);
        left = cat(3, perSubject{:, 1});
        right = cat(3, perSubject{:, 2});
        grand = cat(4, mean(left, 3, "omitnan"), mean(right, 3, "omitnan"), mean(right - left, 3, "omitnan"));
        maps = zeros(3, numel(centers), numel(labels));
        for c = 1:numel(centers)
            inBin = abs(time - centers(c)) <= halfWidth + 1e-9;
            maps(:, c, :) = permute(mean(grand(inBin, :, :), 1), [3, 1, 2]);
        end
        figureHandle = ei.viz.plotTopomapGrid(maps, chanlocs, ...
            RowNames=["Left-hand MI", "Right-hand MI", "Right − Left"], ...
            ColumnNames=compose("%+.1f s", centers), ...
            Title=sprintf("PhysioNet motor imagery, %s band ERD/S (n = %d participants, causal features)", ...
                bandNames(b), numel(subjects)));
        path = fullfile(options.OutputFolder, "erds_topomaps_" + bandNames(b) + ".png");
        exportgraphics(figureHandle, path, Resolution=150);
        close(figureHandle);
        figurePaths(end + 1) = path; %#ok<AGROW>

        panels = ["C3", "Cz", "C4"];
        [~, panelIndex] = ismember(panels, labels);
        series = struct("Name", {"Left-hand MI", "Right-hand MI"}, "Data", ...
            {permute(left(:, panelIndex, :), [3, 1, 2]), permute(right(:, panelIndex, :), [3, 1, 2])});
        figureHandle = ei.viz.plotErdsTimecourse(time, series, PanelNames=panels, ...
            Title=sprintf("%s band ERD/S, mean ± 95%% CI across %d participants", bandNames(b), numel(subjects)));
        path = fullfile(options.OutputFolder, "erds_timecourse_" + bandNames(b) + ".png");
        exportgraphics(figureHandle, path, Resolution=150);
        close(figureHandle);
        figurePaths(end + 1) = path; %#ok<AGROW>
    end

    summary = struct( ...
        "generated", string(datetime("now", Format="yyyy-MM-dd'T'HH:mm:ss")), ...
        "git_commit", ei.util.gitCommit(), ...
        "config_sha256", config.sha256, ...
        "dataset", "PhysioNet EEGMMIDB runs 4, 8, 12 (left/right fist imagery)", ...
        "subjects", subjects, ...
        "skipped_known_defects", intersect(options.Subjects, knownDefects), ...
        "per_subject", records, ...
        "step_time_ms_median", median(stepTimes(:, 1)), ...
        "step_time_ms_p99", max(stepTimes(:, 2)), ...
        "chunk_seconds", config.chunk_seconds, ...
        "figures", figurePaths);
    manifestPath = fullfile(options.OutputFolder, "run_manifest.json");
    writelines(jsonencode(summary, PrettyPrint=true), manifestPath);
    fprintf("Step time per %.0f ms chunk: median %.3f ms, worst p99 %.3f ms\n", 1000*config.chunk_seconds, ...
        summary.step_time_ms_median, summary.step_time_ms_p99);
    fprintf("Outputs written to %s\n", options.OutputFolder);
end

function epochs = mergeEpochs(parts)
    epochs = parts{1};
    for k = 2:numel(parts)
        epochs.Data = cat(1, epochs.Data, parts{k}.Data);
        epochs.Type = [epochs.Type; parts{k}.Type];
        epochs.Run = [epochs.Run; parts{k}.Run];
        epochs.Trial = [epochs.Trial; parts{k}.Trial];
        epochs.OnsetSeconds = [epochs.OnsetSeconds; parts{k}.OnsetSeconds];
    end
end
