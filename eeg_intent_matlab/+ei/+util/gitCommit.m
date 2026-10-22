function commit = gitCommit()
%GITCOMMIT Current git commit of the project, with "-dirty" if there are uncommitted changes.

    root = ei.util.projectRoot();
    [status, output] = system(sprintf('git -C "%s" rev-parse HEAD', root));
    if status ~= 0
        commit = "unavailable";
        return
    end
    commit = strtrim(string(output));
    [status, output] = system(sprintf('git -C "%s" status --porcelain', root));
    if status == 0 && strlength(strtrim(string(output))) > 0
        commit = commit + "-dirty";
    end
end
