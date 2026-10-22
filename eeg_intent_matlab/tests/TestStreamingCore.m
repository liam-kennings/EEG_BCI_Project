classdef TestStreamingCore < matlab.unittest.TestCase
    %TESTSTREAMINGCORE Filter state, band power, block timing, and causality.

    properties
        Config struct
    end

    methods (TestClassSetup)
        function addProjectToPath(testCase)
            root = fileparts(fileparts(mfilename("fullpath")));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture(root));
            testCase.Config = ei.config.loadConfig(fullfile(root, "config", "pipeline_config.json"));
        end
    end

    methods (Test)
        function chunkedFilteringMatchesSingleShot(testCase)
            fs = 160;
            [sections, gain] = ei.dsp.designButterworth("bandpass", [8, 12], 4, fs);
            rng(1);
            x = randn(3000, 5);
            whole = ei.dsp.SosFilter(sections, gain, 5).step(x);
            chunked = ei.dsp.SosFilter(sections, gain, 5);
            edges = [0, sort(randperm(2999, 40)), 3000];
            pieces = cell(numel(edges) - 1, 1);
            for k = 1:numel(edges) - 1
                pieces{k} = chunked.step(x(edges(k) + 1:edges(k + 1), :));
            end
            testCase.verifyEqual(cat(1, pieces{:}), whole, AbsTol=1e-12);
            testCase.verifyEqual(whole, gain*sosfilt(sections, x), AbsTol=1e-10);
        end

        function sineInMuBandRaisesMuNotBeta(testCase)
            fs = 160;
            t = (0:fs*20 - 1)'/fs;
            rng(2);
            x = 10*sin(2*pi*10*t) + randn(numel(t), 4);
            pipeline = ei.pipeline.StreamingPipeline(withReference(testCase.Config, "none"), fs, ...
                ["C3", "Cz", "C4", "Pz"]);
            rows = pipeline.step(x);
            warm = rows.IsFilterWarm;
            muMinusBeta = 10*mean(rows.LogPower(warm, :, 1) - rows.LogPower(warm, :, 2), "all");
            testCase.verifyGreaterThan(muMinusBeta, 15);
        end

        function blockTimingHandlesFractionalBlockLengths(testCase)
            for fs = [160, 250, 512]
                stream = ei.stream.BandPowerStream(fs, 2, 1, BlockSeconds=0.25);
                sampleCount = round(fs*10.3);
                rows = stream.step(ones(sampleCount, 2, 1), ones(sampleCount, 2));
                testCase.verifyEqual(numel(rows.Time), floor(sampleCount/(0.25*fs)));
                testCase.verifyEqual(rows.Time, (1:numel(rows.Time))'*0.25, AbsTol=1/fs);
            end
        end

        function smoothingAndLogMatchHandCalculation(testCase)
            fs = 100;
            power = [ones(25, 1); 4*ones(25, 1); 16*ones(25, 1)];
            stream = ei.stream.BandPowerStream(fs, 1, 1, BlockSeconds=0.25, SmoothingBlocks=3, DriftEnabled=false);
            rows = stream.step(reshape(power, [], 1, 1), zeros(numel(power), 1));
            expected = log10([1; (1 + 4)/2; (1 + 4 + 16)/3]);
            testCase.verifyEqual(rows.LogPower(:), expected, AbsTol=1e-12);
        end

        function driftCompensationRemovesStationaryLevel(testCase)
            fs = 100;
            stream = ei.stream.BandPowerStream(fs, 1, 1, DriftSeconds=5, DriftEnabled=true);
            rows = stream.step(7*ones(fs*20, 1, 1), zeros(fs*20, 1));
            testCase.verifyEqual(rows.LogPowerDrift(rows.IsDriftWarm), zeros(nnz(rows.IsDriftWarm), 1), AbsTol=1e-12);
        end

        function futureSamplesNeverChangePastFeatures(testCase)
            fs = 160;
            rng(3);
            x = randn(fs*30, 6);
            labels = ["FC3", "C3", "Cz", "C4", "FC4", "Pz"];
            original = ei.pipeline.StreamingPipeline(testCase.Config, fs, labels).step(x);
            cut = fs*12;
            modified = x;
            modified(cut + 1:end, :) = 100*randn(size(x, 1) - cut, 6);
            changed = ei.pipeline.StreamingPipeline(testCase.Config, fs, labels).step(modified);
            past = original.Time <= cut/fs;
            testCase.verifyEqual(changed.LogPower(past, :, :), original.LogPower(past, :, :), AbsTol=1e-12);
            testCase.verifyEqual(changed.LogPowerDrift(past, :, :), original.LogPowerDrift(past, :, :), AbsTol=1e-12);
            testCase.verifyEqual(changed.Artifact(past, :), original.Artifact(past, :), AbsTol=1e-12);
            testCase.verifyNotEqual(changed.LogPower(~past, :, :), original.LogPower(~past, :, :));
        end

        function replayEqualsSingleStep(testCase)
            fs = 160;
            rng(4);
            recording = struct("Data", randn(fs*15, 4), "Fs", fs, "Labels", ["C3", "Cz", "C4", "Pz"], ...
                "SourcePath", "synthetic");
            stream = ei.pipeline.replayRecording(recording, testCase.Config);
            direct = ei.pipeline.StreamingPipeline(testCase.Config, fs, recording.Labels).step(recording.Data);
            testCase.verifyEqual(stream.LogPower, direct.LogPower, AbsTol=1e-12);
            testCase.verifyEqual(stream.Time, direct.Time, AbsTol=1e-12);
        end

        function badChannelsAreNaNAndExcludedFromReference(testCase)
            fs = 160;
            rng(5);
            x = randn(fs*10, 4);
            isGood = [true, true, false, true];
            noisy = x;
            noisy(:, 3) = 1e4*randn(fs*10, 1);
            clean = ei.pipeline.StreamingPipeline(testCase.Config, fs, ["C3", "Cz", "C4", "Pz"], IsGood=isGood).step(x);
            dirty = ei.pipeline.StreamingPipeline(testCase.Config, fs, ["C3", "Cz", "C4", "Pz"], IsGood=isGood).step(noisy);
            testCase.verifyTrue(all(isnan(dirty.LogPower(:, 3, :)), "all"));
            testCase.verifyEqual(dirty.LogPower(:, isGood, :), clean.LogPower(:, isGood, :), AbsTol=1e-12);
        end
    end
end

function config = withReference(config, method)
    config.reference.method = method;
    config.drift_compensation.enabled = false;
end
