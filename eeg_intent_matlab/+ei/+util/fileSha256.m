function digest = fileSha256(filePath)
%FILESHA256 SHA-256 hex digest of a file, or "unavailable" if it cannot be computed.
%   Uses the system shasum (macOS/Linux) or certutil (Windows) so the
%   digest matches what Python's hashlib reports for the same file.

    arguments
        filePath (1,1) string
    end

    digest = "unavailable";
    if ispc()
        [status, output] = system(sprintf('certutil -hashfile "%s" SHA256', filePath));
        lines = splitlines(string(output));
        if status == 0 && numel(lines) >= 2
            digest = lower(erase(strtrim(lines(2)), " "));
        end
    else
        [status, output] = system(sprintf('shasum -a 256 "%s"', filePath));
        if status == 0
            parts = split(strtrim(string(output)));
            digest = parts(1);
        end
    end
end
