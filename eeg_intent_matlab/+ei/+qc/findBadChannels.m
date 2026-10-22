function result = findBadChannels(data, fs, config)
%FINDBADCHANNELS Calibration-time bad-channel rule (after Lee et al. 2022).
%   result = ei.qc.findBadChannels(data, fs, config)
%   data    [samples x channels] in microvolts, one run
%   config  pipeline config (uses config.channel_health)
%
%   A channel is bad if it is
%     flat:      standard deviation below flat_std_uv
%     saturated: any |sample| above saturation_uv
%     noisy:     z-score of its log band power (common-average referenced,
%                band_hz, mean removed) above z_threshold, computed over the
%                channels that are not flat or saturated.
%   With robust_z = true (default) the z-score uses the median and MAD.
%   A plain z-score (Lee et al.) cannot exceed (n-1)/sqrt(n) for a single
%   outlier among n channels: 5.4 for 31 channels, 7.9 for 64. So z > 6
%   can never flag one bad channel on a 32-channel montage. Lee et al. had
%   256 channels, where the ceiling is 15.9.
%   Returns a struct: IsBad, IsFlat, IsSaturated, IsNoisy, LogPowerZ.
%   Run this on calibration data only; it is not used on the decision path.

    arguments
        data (:,:) double
        fs (1,1) double {mustBePositive}
        config (1,1) struct
    end

    health = config.channel_health;
    isFlat = std(data, 0, 1) < health.flat_std_uv;
    isSaturated = max(abs(data), [], 1) > health.saturation_uv;
    candidates = ~(isFlat | isSaturated);

    logPowerZ = nan(1, size(data, 2));
    isNoisy = false(1, size(data, 2));
    if nnz(candidates) >= 3
        referenced = data*ei.stream.rereferenceMatrix(candidates, "average");
        [sections, gain] = ei.dsp.designButterworth("bandpass", reshape(health.band_hz, 1, []), config.filter_order, fs);
        filtered = ei.dsp.SosFilter(sections, gain, size(data, 2)).step(referenced);
        logPower = log(mean((filtered - mean(filtered, 1)).^2, 1));
        logPowerZ(candidates) = standardize(logPower(candidates), health.robust_z);
        isNoisy = logPowerZ > health.z_threshold;
    end

    result = struct("IsBad", isFlat | isSaturated | isNoisy, "IsFlat", isFlat, ...
        "IsSaturated", isSaturated, "IsNoisy", isNoisy, "LogPowerZ", logPowerZ);
end

function z = standardize(values, isRobust)
    if isRobust
        scale = 1.4826*median(abs(values - median(values)));
        z = (values - median(values))/max(scale, eps);
    else
        z = zscore(values);
    end
end
