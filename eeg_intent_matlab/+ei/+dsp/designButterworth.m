function [sections, gain] = designButterworth(kind, edgesHz, order, fs)
%DESIGNBUTTERWORTH Butterworth filter as second-order sections.
%   [sections, gain] = ei.dsp.designButterworth(kind, edgesHz, order, fs)
%   kind     "bandpass", "stop", "low", or "high"
%   edgesHz  [low high] for bandpass/stop, scalar for low/high
%   order    design order passed to butter (a bandpass of order N has 2N poles)
%   fs       sampling rate in Hz

    arguments
        kind (1,1) string {mustBeMember(kind, ["bandpass", "stop", "low", "high"])}
        edgesHz (1,:) double {mustBePositive}
        order (1,1) double {mustBePositive, mustBeInteger}
        fs (1,1) double {mustBePositive}
    end

    nyquist = fs/2;
    if any(edgesHz >= nyquist)
        error("ei:dsp:edges", "Filter edges %s Hz must be below the Nyquist frequency (%.1f Hz).", ...
            mat2str(edgesHz), nyquist);
    end
    [z, p, k] = butter(order, edgesHz/nyquist, kind);
    [sections, gain] = zp2sos(z, p, k);
end
