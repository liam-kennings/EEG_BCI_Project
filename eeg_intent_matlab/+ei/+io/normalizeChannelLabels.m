function labels = normalizeChannelLabels(rawLabels)
%NORMALIZECHANNELLABELS Convert raw EEG channel names to standard 10-05 spelling.
%   "Fc3." -> "FC3", "Fpz." -> "Fpz", "Cz.." -> "Cz", "EEG C3-REF" -> "C3".
%   Rules: drop dots, an "EEG " prefix, and a "-REF"/"-LE" suffix; upper-case;
%   then a leading "FP" becomes "Fp" and a trailing "Z" becomes "z".

    arguments
        rawLabels string
    end

    labels = strtrim(erase(rawLabels, "."));
    labels = regexprep(labels, "^(?i)EEG\s+", "");
    labels = regexprep(labels, "(?i)-(REF|LE|AVG)$", "");
    labels = upper(labels);
    labels = regexprep(labels, "^FP", "Fp");
    labels = regexprep(labels, "Z$", "z");
    labels = reshape(labels, 1, []);
end
