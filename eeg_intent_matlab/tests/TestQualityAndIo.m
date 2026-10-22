classdef TestQualityAndIo < matlab.unittest.TestCase
    %TESTQUALITYANDIO Re-referencing, bad-channel rules, labels, events, and config.

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
        function averageReferenceSumsToZeroOverGoodChannels(testCase)
            isGood = [true, true, false, true, true];
            weights = ei.stream.rereferenceMatrix(isGood, "average");
            rng(6);
            referenced = randn(100, 5)*weights;
            testCase.verifyEqual(sum(referenced(:, isGood), 2), zeros(100, 1), AbsTol=1e-12);
            testCase.verifyEqual(referenced(:, 3), zeros(100, 1));
        end

        function laplacianSubtractsNearestNeighbours(testCase)
            positions = [0, 0, 0; 1, 0, 0; -1, 0, 0; 0, 1, 0; 0, -1, 0; 5, 5, 0];
            weights = ei.stream.rereferenceMatrix(true(1, 6), "laplacian", positions, 4);
            testCase.verifyEqual(weights(:, 1)', [1, -0.25, -0.25, -0.25, -0.25, 0], AbsTol=1e-12);
        end

        function flatSaturatedAndNoisyChannelsAreFlagged(testCase)
            fs = 160;
            rng(7);
            data = 10*randn(fs*20, 32);
            data(:, 4) = 0;
            data(:, 9) = 1500*sign(randn(fs*20, 1));
            data(:, 20) = 150*randn(fs*20, 1);
            result = ei.qc.findBadChannels(data, fs, testCase.Config);
            testCase.verifyTrue(result.IsFlat(4));
            testCase.verifyTrue(result.IsSaturated(9));
            testCase.verifyTrue(result.IsNoisy(20));
            testCase.verifyEqual(find(result.IsBad), [4, 9, 20]);
        end

        function plainZScoreCannotFlagOneOfThirtyTwoChannels(testCase)
            % Documents why robust_z is the default: (n-1)/sqrt(n) < 6 for n < 38.
            fs = 160;
            rng(8);
            data = 10*randn(fs*20, 32);
            data(:, 20) = 150*randn(fs*20, 1);
            config = testCase.Config;
            config.channel_health.robust_z = false;
            result = ei.qc.findBadChannels(data, fs, config);
            testCase.verifyFalse(result.IsNoisy(20));
            testCase.verifyLessThan(max(result.LogPowerZ), 31/sqrt(32) + 1e-9);
        end

        function runQualityPoolsBadChannelsOverKeptRuns(testCase)
            isBadByRun = false(3, 20);
            isBadByRun(1, 2) = true;
            isBadByRun(3, 1:5) = true;
            [isGoodChannel, isGoodRun] = ei.qc.combineRunQuality(isBadByRun, 0.1);
            testCase.verifyEqual(isGoodRun, [true; true; false]);
            testCase.verifyEqual(find(~isGoodChannel), 2);
        end

        function labelsAreNormalizedToStandardSpelling(testCase)
            raw = ["Fc3.", "Fpz.", "Cz..", "Afz.", "EEG C4-REF", "Iz.."];
            testCase.verifyEqual(ei.io.normalizeChannelLabels(raw), ["FC3", "Fpz", "Cz", "AFz", "C4", "Iz"]);
        end

        function physionetCodesMapByRun(testCase)
            annotations = table([0; 4.2; 8.3], [4.2; 4.1; 4.2], ["T0"; "T2"; "T1"], ...
                VariableNames=["OnsetSeconds", "DurationSeconds", "Label"]);
            imagery = ei.io.physionetEvents(annotations, 4);
            testCase.verifyEqual(imagery.Type, ["rest_start"; "cue_right"; "cue_left"]);
            feet = ei.io.physionetEvents(annotations, 6);
            testCase.verifyEqual(feet.Type, ["rest_start"; "cue_both_feet"; "cue_both_fists"]);
        end

        function configRecordsItsHash(testCase)
            testCase.verifyEqual(strlength(testCase.Config.sha256), 64);
        end

        function epochingAndErdsRecoverAKnownDrop(testCase)
            rate = 4;
            time = (1:400)'/rate;
            logPower = zeros(400, 2, 1);
            onsets = [20; 40; 60];
            for onset = onsets'
                during = time > onset + 0.5 & time <= onset + 2;
                logPower(during, 1, 1) = -0.3;
            end
            stream = struct("Time", time, "LogPower", logPower, "LogPowerDrift", logPower, ...
                "IsFilterWarm", true(400, 1), "FeatureRateHz", rate, "Labels", ["C3", "C4"], "BandNames", "mu");
            events = table(onsets, ones(3, 1), repmat("cue_right", 3, 1), [4; 4; 4], (1:3)', ...
                VariableNames=["OnsetSeconds", "DurationSeconds", "Type", "Run", "Trial"]);
            epochs = ei.analysis.epochFeatures(stream, events, "cue_right", TminSeconds=-2, TmaxSeconds=4);
            erds = ei.analysis.computeErds(epochs, [-1, 0]);
            testCase.verifySize(erds.Data, [3, numel(-2:0.25:4), 2, 1]);
            atOneSecond = erds.Time == 1;
            testCase.verifyEqual(squeeze(erds.Data(:, atOneSecond, 1, 1)), -3*ones(3, 1), AbsTol=1e-12);
            testCase.verifyEqual(squeeze(erds.Data(:, atOneSecond, 2, 1)), zeros(3, 1), AbsTol=1e-12);
        end
    end
end
