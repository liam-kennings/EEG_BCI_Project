function figureHandle = plotErdsTimecourse(time, series, options)
%PLOTERDSTIMECOURSE ERD/S time courses with 95% confidence bands, one panel per channel.
%   figureHandle = ei.viz.plotErdsTimecourse(time, series, Name=Value)
%   time    [1 x T] seconds relative to the cue
%   series  struct array, one element per condition, with fields
%             Name   condition name
%             Data   [units x T x panels] values per participant (or trial)
%   Name=Value: PanelNames, Title, Units ("dB"), Visible ("off")
%   Bands show mean +/- 1.96 standard errors across units.

    arguments
        time (1,:) double
        series struct
        options.PanelNames (1,:) string = strings(1, 0)
        options.Title (1,1) string = ""
        options.Units (1,1) string = "dB"
        options.Visible (1,1) string {mustBeMember(options.Visible, ["on", "off"])} = "off"
    end

    palette = [42, 120, 214; 235, 104, 52; 27, 175, 122]/255;
    panelCount = size(series(1).Data, 3);
    figureHandle = figure(Visible=options.Visible, Color="w", Position=[100, 100, 380*panelCount, 320]);
    layout = tiledlayout(figureHandle, 1, panelCount, TileSpacing="compact", Padding="compact");
    for p = 1:panelCount
        axesHandle = nexttile(layout);
        hold(axesHandle, "on");
        yline(axesHandle, 0, Color=[0.55, 0.55, 0.55], LineStyle="--", HandleVisibility="off");
        xline(axesHandle, 0, Color=[0.55, 0.55, 0.55], LineStyle=":", HandleVisibility="off");
        for s = 1:numel(series)
            values = series(s).Data(:, :, p);
            center = mean(values, 1, "omitnan");
            spread = 1.96*std(values, 0, 1, "omitnan")./sqrt(sum(~isnan(values), 1));
            color = palette(mod(s - 1, size(palette, 1)) + 1, :);
            fill(axesHandle, [time, fliplr(time)], [center - spread, fliplr(center + spread)], color, ...
                FaceAlpha=0.15, EdgeColor="none", HandleVisibility="off");
            plot(axesHandle, time, center, Color=color, LineWidth=2, DisplayName=series(s).Name);
        end
        if p <= numel(options.PanelNames)
            title(axesHandle, options.PanelNames(p), FontWeight="normal");
        end
        xlabel(axesHandle, "Time from cue (s)");
        if p == 1
            ylabel(axesHandle, "ERD/S (" + options.Units + ")");
            legend(axesHandle, Location="southwest", Box="off");
        end
        grid(axesHandle, "on");
        axesHandle.GridColor = [0.85, 0.85, 0.85];
        box(axesHandle, "off");
    end
    linkaxes(findobj(figureHandle, "Type", "axes"), "y");
    if strlength(options.Title) > 0
        title(layout, options.Title, FontWeight="bold", FontSize=12);
    end
end
