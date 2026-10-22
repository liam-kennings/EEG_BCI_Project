function figureHandle = plotTopomapGrid(maps, chanlocs, options)
%PLOTTOPOMAPGRID Grid of scalp topographies sharing one symmetric colour scale.
%   figureHandle = ei.viz.plotTopomapGrid(maps, chanlocs, Name=Value)
%   maps      [rows x columns x channels]; NaN channels are left out of each map
%   chanlocs  EEGLAB chanlocs in channel order (ei.io.channelLocations)
%   Name=Value:
%     RowNames, ColumnNames  labels for rows (conditions) and columns (time bins)
%     Title, Units           figure title and colour-bar label (default "dB")
%     ColorLimit             symmetric limit; default = 98th percentile of |maps|
%     Visible                "off" (default) or "on"
%   Requires EEGLAB topoplot. Axes are placed manually because topoplot
%   resizes its axes, which conflicts with tiledlayout.

    arguments
        maps (:,:,:) double
        chanlocs struct
        options.RowNames (1,:) string = strings(1, 0)
        options.ColumnNames (1,:) string = strings(1, 0)
        options.Title (1,1) string = ""
        options.Units (1,1) string = "dB"
        options.ColorLimit (1,1) double = NaN
        options.Visible (1,1) string {mustBeMember(options.Visible, ["on", "off"])} = "off"
    end

    [rowCount, columnCount, ~] = size(maps);
    limit = options.ColorLimit;
    if isnan(limit)
        limit = prctile(abs(maps(:)), 98);
    end
    if ~(limit > 0)
        limit = 1;
    end
    colors = ei.viz.divergingColormap(256);
    hasLocation = arrayfun(@(location) ~isempty(location.theta), chanlocs);

    tileSize = 150;
    leftMargin = 130;
    rightMargin = 110;
    topMargin = 70;
    bottomMargin = 15;
    width = leftMargin + tileSize*columnCount + rightMargin;
    height = topMargin + tileSize*rowCount + bottomMargin;
    figureHandle = figure(Visible=options.Visible, Color="w", Units="pixels", Position=[100, 100, width, height]);
    colormap(figureHandle, colors);
    for r = 1:rowCount
        for c = 1:columnCount
            position = [leftMargin + (c - 1)*tileSize, height - topMargin - r*tileSize, tileSize, tileSize]./[width, height, width, height];
            axesHandle = axes(figureHandle, Position=position);
            values = squeeze(maps(r, c, :))';
            keep = hasLocation & ~isnan(values);
            % EEGLAB parses options with strcmp on char arrays, so they are not strings here.
            topoplot(values(keep), chanlocs(keep), 'maplimits', [-limit, limit], 'electrodes', 'on', ...
                'numcontour', 0, 'colormap', colors, 'whitebk', 'on', 'emarker', {'.', [0.25, 0.25, 0.25], 4, 1});
            clim(axesHandle, [-limit, limit]);
            if r == 1 && c <= numel(options.ColumnNames)
                title(axesHandle, options.ColumnNames(c), FontWeight="normal", FontSize=10);
            end
        end
        if r <= numel(options.RowNames)
            annotation(figureHandle, "textbox", [0, (height - topMargin - (r - 0.5)*tileSize - 15)/height, ...
                (leftMargin - 10)/width, 30/height], String=options.RowNames(r), HorizontalAlignment="right", ...
                VerticalAlignment="middle", EdgeColor="none", FontSize=10);
        end
    end
    barAxes = axes(figureHandle, Position=[(width - rightMargin + 25)/width, (bottomMargin + 0.15*tileSize*rowCount)/height, ...
        0.001, 0.7*tileSize*rowCount/height], Visible="off");
    colormap(barAxes, colors);
    clim(barAxes, [-limit, limit]);
    bar = colorbar(barAxes, Position=barAxes.Position + [0, 0, 18/width, 0]);
    bar.Label.String = options.Units;
    if strlength(options.Title) > 0
        annotation(figureHandle, "textbox", [0.01, (height - 40)/height, 0.98, 35/height], String=options.Title, ...
            EdgeColor="none", FontWeight="bold", FontSize=12, HorizontalAlignment="left");
    end
end
