classdef SosFilter < handle
    %SOSFILTER Causal second-order-section IIR filter that keeps state between chunks.
    %   MATLAB's sosfilt has no state argument, so each section is run with
    %   filter() and its final conditions are carried to the next call.
    %   Filtering a signal in chunks gives the same output as filtering it
    %   in one call.
    %
    %   f = ei.dsp.SosFilter(sections, gain, channelCount)
    %   y = f.step(x)    x and y are [samples x channels]
    %   f.reset()        clear the state

    properties (SetAccess = private)
        Sections (:,6) double
        Gain (1,1) double
        ChannelCount (1,1) double
    end

    properties (Access = private)
        State cell
    end

    methods
        function obj = SosFilter(sections, gain, channelCount)
            arguments
                sections (:,6) double
                gain (1,1) double
                channelCount (1,1) double {mustBePositive, mustBeInteger}
            end
            obj.Sections = sections;
            obj.Gain = gain;
            obj.ChannelCount = channelCount;
            obj.reset();
        end

        function y = step(obj, x)
            %STEP Filter one chunk [samples x channels] and update the state.
            arguments
                obj
                x (:,:) double
            end
            if size(x, 2) ~= obj.ChannelCount
                error("ei:dsp:channels", "Expected %d channels but the chunk has %d.", ...
                    obj.ChannelCount, size(x, 2));
            end
            y = x;
            if isempty(x)
                return
            end
            for k = 1:size(obj.Sections, 1)
                [y, obj.State{k}] = filter(obj.Sections(k, 1:3), obj.Sections(k, 4:6), y, obj.State{k}, 1);
            end
            y = obj.Gain*y;
        end

        function reset(obj)
            %RESET Clear the filter state (zero initial conditions).
            obj.State = repmat({zeros(2, obj.ChannelCount)}, size(obj.Sections, 1), 1);
        end
    end
end
