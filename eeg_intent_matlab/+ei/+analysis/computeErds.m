function erds = computeErds(epochs, referenceWindowSeconds)
%COMPUTEERDS Event-related (de)synchronization in dB relative to a pre-cue reference.
%   erds = ei.analysis.computeErds(epochs, referenceWindowSeconds)
%   epochs  struct from ei.analysis.epochFeatures with Field "LogPower" (log10 power)
%   referenceWindowSeconds  [start end] relative to the event, e.g. [-1 0]
%
%   ERD/S(t) = 10 * (log10 P(t) - mean over the reference window of log10 P),
%   per trial, channel, and band. Negative values are desynchronization (ERD).
%   Returns the epochs struct with Data in dB and Units = "dB".

    arguments
        epochs (1,1) struct
        referenceWindowSeconds (1,2) double
    end

    inReference = epochs.Time >= referenceWindowSeconds(1) - 1e-9 & epochs.Time <= referenceWindowSeconds(2) + 1e-9;
    if ~any(inReference)
        error("ei:analysis:reference", "No epoch samples fall inside the reference window %s s.", ...
            mat2str(referenceWindowSeconds));
    end
    reference = mean(epochs.Data(:, inReference, :, :), 2);
    erds = epochs;
    erds.Data = 10*(epochs.Data - reference);
    erds.Units = "dB";
    erds.ReferenceWindowSeconds = referenceWindowSeconds;
end
