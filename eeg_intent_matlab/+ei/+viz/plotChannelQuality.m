function figureHandle = plotChannelQuality(isBadByRun, labels, runNames, options)
%PLOTCHANNELQUALITY Channels x runs grid of kept (dark) and rejected (red) channels.
%   figureHandle = ei.viz.plotChannelQuality(isBadByRun, labels, runNames, Name=Value)
%   isBadByRun  [runs x channels] logical
%   Name=Value: IsGoodRun [runs x 1] logical (rejected runs are shaded), Title, Visible

    arguments
        isBadByRun (:,:) logical
        labels (1,:) string
        runNames (1,:) string
        options.IsGoodRun (:,1) logical = true(size(isBadByRun, 1), 1)
        options.Title (1,1) string = "Channel quality"
        options.Visible (1,1) string {mustBeMember(options.Visible, ["on", "off"])} = "off"
    end

    [runCount, channelCount] = size(isBadByRun);
    figureHandle = figure(Visible=options.Visible, Color="w", Position=[100, 100, 14*channelCount + 160, 40*runCount + 140]);
    axesHandle = axes(figureHandle);
    hold(axesHandle, "on");
    for r = find(~options.IsGoodRun)'
        patch(axesHandle, [0.5, channelCount + 0.5, channelCount + 0.5, 0.5], [r - 0.5, r - 0.5, r + 0.5, r + 0.5], ...
            [0.95, 0.85, 0.85], EdgeColor="none");
    end
    [badRun, badChannel] = find(isBadByRun);
    [goodRun, goodChannel] = find(~isBadByRun);
    scatter(axesHandle, goodChannel, goodRun, 18, [0.2, 0.2, 0.2], "filled", DisplayName="kept");
    scatter(axesHandle, badChannel, badRun, 30, [0.89, 0.29, 0.28], "filled", DisplayName="rejected");
    axesHandle.YDir = "reverse";
    axesHandle.XTick = 1:channelCount;
    axesHandle.XTickLabel = labels;
    axesHandle.XTickLabelRotation = 90;
    axesHandle.YTick = 1:runCount;
    axesHandle.YTickLabel = runNames;
    axesHandle.FontSize = 8;
    xlim(axesHandle, [0.5, channelCount + 0.5]);
    ylim(axesHandle, [0.5, runCount + 0.5]);
    legend(axesHandle, Location="northoutside", Orientation="horizontal", Box="off");
    title(axesHandle, options.Title, FontWeight="normal");
end
