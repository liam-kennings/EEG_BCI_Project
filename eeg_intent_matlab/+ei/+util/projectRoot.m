function root = projectRoot()
%PROJECTROOT Absolute path of the eeg_intent_matlab folder.

    root = string(fileparts(fileparts(fileparts(mfilename("fullpath")))));
end
