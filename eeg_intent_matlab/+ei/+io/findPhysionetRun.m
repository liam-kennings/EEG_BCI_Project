function filePath = findPhysionetRun(root, subject, run)
%FINDPHYSIONETRUN Locate SxxxRyy.edf under an EEGMMIDB root (handles nested folders).

    arguments
        root (1,1) string {mustBeFolder}
        subject (1,1) double {mustBeInteger, mustBePositive}
        run (1,1) double {mustBeInteger, mustBePositive}
    end

    fileName = sprintf("S%03dR%02d.edf", subject, run);
    candidates = dir(fullfile(root, sprintf("S%03d", subject), "**", fileName));
    if isempty(candidates)
        error("ei:io:physionetMissing", "%s not found under %s. Check data.physionet_root in the config.", ...
            fileName, root);
    end
    filePath = string(fullfile(candidates(1).folder, candidates(1).name));
end
