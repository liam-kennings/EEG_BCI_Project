function [chanlocs, positions, isFound] = channelLocations(labels)
%CHANNELLOCATIONS Standard 10-05 electrode locations for the given labels (EEGLAB).
%   [chanlocs, positions, isFound] = ei.io.channelLocations(labels)
%   chanlocs   EEGLAB chanlocs struct in label order (for topoplot)
%   positions  [channels x 3] Cartesian coordinates (for Laplacian reference)
%   isFound    logical, false for labels missing from the template
%   Requires EEGLAB (uses its Standard-10-5-Cap385 template and readlocs).

    arguments
        labels (1,:) string
    end

    persistent template templateLabels
    if isempty(template)
        if isempty(which("eeglab")) || isempty(which("readlocs"))
            error("ei:io:eeglab", "EEGLAB is not on the path. Add it (or run eeglab nogui) for electrode locations.");
        end
        templateFile = fullfile(fileparts(which("eeglab")), "functions", "supportfiles", ...
            "Standard-10-5-Cap385_witheog.elp");
        template = readlocs(char(templateFile), 'filetype', 'besa');  % EEGLAB expects char options
        templateLabels = upper(string({template.labels}));
    end

    [isFound, index] = ismember(upper(labels), templateLabels);
    chanlocs = template(max(index, 1));
    positions = nan(numel(labels), 3);
    for k = find(isFound)
        chanlocs(k).labels = char(labels(k));
        positions(k, :) = [template(index(k)).X, template(index(k)).Y, template(index(k)).Z];
    end
    for k = find(~isFound)
        chanlocs(k).labels = char(labels(k));
        chanlocs(k).theta = [];
        chanlocs(k).radius = [];
    end
end
