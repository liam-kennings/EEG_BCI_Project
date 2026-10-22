function plan = buildfile
%BUILDFILE Build tasks for the MATLAB motor-intent pipeline.
%   buildtool          run the Code Analyzer and the unit tests
%   buildtool check    Code Analyzer only
%   buildtool test     unit tests only
%   buildtool demo     PhysioNet end-to-end demo (needs data and EEGLAB)

    plan = buildplan(localfunctions);
    plan("test").Dependencies = "check";
    plan.DefaultTasks = "test";
end

function checkTask(~)
    % Fail on any Code Analyzer issue in the package and tests
    issues = codeIssues(["+ei", "tests", "demos"]);
    if height(issues.Issues) > 0
        disp(issues.Issues(:, ["Location", "Severity", "Description"]));
        error("ei:build:codeIssues", "Fix the %d Code Analyzer issue(s) listed above.", height(issues.Issues));
    end
end

function testTask(~)
    % Run the unit tests
    results = runtests("tests");
    disp(table(results));
    assertSuccess(results);
end

function demoTask(~)
    % Run the PhysioNet end-to-end demo
    addpath(fullfile(pwd, "demos"));
    runPhysionetDemo();
end
