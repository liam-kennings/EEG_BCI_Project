function events = physionetEvents(annotations, run)
%PHYSIONETEVENTS Map EEGMMIDB T0/T1/T2 annotations to pipeline event types.
%   events = ei.io.physionetEvents(annotations, run)
%   annotations  table from ei.io.readEdfRecording
%   run          EEGMMIDB run number (3-14)
%
%   Runs 3/7/11 (executed) and 4/8/12 (imagined) are left/right fist:
%     T1 -> cue_left, T2 -> cue_right.
%   Runs 5/9/13 (executed) and 6/10/14 (imagined) are both fists/both feet:
%     T1 -> cue_both_fists, T2 -> cue_both_feet.
%   T0 -> rest_start in every run.
%   Returns a table: OnsetSeconds, DurationSeconds, Type, Run, Trial.

    arguments
        annotations table
        run (1,1) double {mustBeInteger, mustBeBetween(run, 3, 14)}
    end

    if any(run == [3, 4, 7, 8, 11, 12])
        taskNames = ["cue_left", "cue_right"];
    else
        taskNames = ["cue_both_fists", "cue_both_feet"];
    end
    codes = upper(strtrim(annotations.Label));
    type = strings(height(annotations), 1);
    type(codes == "T0") = "rest_start";
    type(codes == "T1") = taskNames(1);
    type(codes == "T2") = taskNames(2);
    keep = type ~= "";

    events = table(annotations.OnsetSeconds(keep), annotations.DurationSeconds(keep), type(keep), ...
        repmat(run, nnz(keep), 1), (1:nnz(keep))', ...
        VariableNames=["OnsetSeconds", "DurationSeconds", "Type", "Run", "Trial"]);
end
