function config = loadConfig(configPath)
%LOADCONFIG Read and validate the shared pipeline configuration.
%   config = ei.config.loadConfig() reads config/pipeline_config.json.
%   config = ei.config.loadConfig(configPath) reads another file.
%
%   The returned struct keeps the JSON field names (snake_case) so MATLAB
%   and Python read identical keys. Two fields are added:
%     config.source_path  absolute path of the file that was read
%     config.sha256       SHA-256 of that file, recorded in every output

    arguments
        configPath (1,1) string = fullfile(ei.util.projectRoot(), "config", "pipeline_config.json")
    end

    if ~isfile(configPath)
        error("ei:config:missing", "Configuration file not found: %s. Pass a valid path.", configPath);
    end
    config = jsondecode(fileread(configPath));
    validateConfig(config);
    config.source_path = string(configPath);
    config.sha256 = ei.util.fileSha256(configPath);
end

function validateConfig(config)
    required = ["schema_version", "chunk_seconds", "filter_warmup_seconds", "filter_order", ...
        "line_noise", "reference", "bands", "broadband", "band_power", "drift_compensation", ...
        "channel_health", "epoch", "erds", "topomap"];
    missing = required(~isfield(config, required));
    if ~isempty(missing)
        error("ei:config:fields", "Configuration is missing fields: %s.", strjoin(missing, ", "));
    end
    for band = reshape(config.bands, 1, [])
        if ~(band.low_hz > 0 && band.high_hz > band.low_hz)
            error("ei:config:band", "Band '%s' needs 0 < low_hz < high_hz.", band.name);
        end
    end
    validMethods = ["average", "laplacian", "none"];
    if ~any(config.reference.method == validMethods)
        error("ei:config:reference", "reference.method must be one of: %s.", strjoin(validMethods, ", "));
    end
    if config.band_power.block_seconds <= 0 || config.band_power.smoothing_blocks < 1
        error("ei:config:bandPower", "band_power needs block_seconds > 0 and smoothing_blocks >= 1.");
    end
end
