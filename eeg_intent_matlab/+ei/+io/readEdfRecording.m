function recording = readEdfRecording(filePath)
%READEDFRECORDING Read an EDF/EDF+ file into the pipeline's recording struct.
%   recording = ei.io.readEdfRecording(filePath) returns:
%     Data         [samples x channels] double, microvolts
%     Fs           sampling rate (Hz)
%     Labels       [1 x channels] string, standard 10-05 spelling
%     Annotations  table: OnsetSeconds, DurationSeconds, Label
%     SourcePath   absolute file path
%   All signals must share one sampling rate.

    arguments
        filePath (1,1) string {mustBeFile}
    end

    info = edfinfo(filePath);
    if numel(unique(info.NumSamples)) ~= 1
        error("ei:io:mixedRates", "%s mixes sampling rates; select channels with one rate before reading.", filePath);
    end
    [records, annotations] = edfread(filePath);
    data = cell2mat(records{:, :});
    recordSeconds = seconds(info.DataRecordDuration);
    fs = double(info.NumSamples(1))/recordSeconds;

    units = string(info.PhysicalDimensions);
    scale = ones(1, numel(units));
    scale(strcmpi(units, "V")) = 1e6;
    scale(strcmpi(units, "mV")) = 1e3;
    scale(strcmpi(units, "nV")) = 1e-3;

    recording = struct( ...
        "Data", double(data).*scale, ...
        "Fs", fs, ...
        "Labels", ei.io.normalizeChannelLabels(string(info.SignalLabels)), ...
        "Annotations", table(seconds(annotations.Onset), seconds(annotations.Duration), ...
            string(annotations.Annotations), VariableNames=["OnsetSeconds", "DurationSeconds", "Label"]), ...
        "SourcePath", string(filePath));
end
