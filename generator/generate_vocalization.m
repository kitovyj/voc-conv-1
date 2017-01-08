function voc = generate_vocalizations(varargin)
        
    P = parsePairs(varargin);
    checkField(P, 'vibrato_amp', 1000);
    checkField(P, 'peaks_num', 1);        

    % all frequencies are in Hz
    freq_min = 40000; 
    freq_deviation = 20000;
    peak_deviation = 7000;
    
    % all times are in s    
    start_time_min = 0.02;
    start_time_deviation = 0.01;
    duration_min = 0.08;
    duration_deviation = 0.02;
    vibrato_period = 0.01;
    
    start_time = start_time_min + 2*rand()*start_time_deviation;
    
    duration = duration_min + 2*duration_deviation*rand();
        
    start_freq = freq_min + 2*rand()*freq_deviation;
    end_freq = freq_min + 2*rand()*freq_deviation;
    
    t = linspace(0, 1, 1000);
        
    peak_min_amp = max(start_freq, end_freq) + 5000;
    peak_max_amp = max(peak_min_amp, 80000);
    %peak_min_amp + peak_deviation;
    peaks = generate_peaks(start_time, start_freq, start_time + duration, end_freq, peak_min_amp, peak_max_amp, P.peaks_num);
        
    start_point = [start_time; start_freq];
    end_point = [start_time + duration; end_freq ];
    
    nodes = [ start_point ];
    
    for i = 1:size(peaks, 2)
        
        curr = peaks(:, i);
        
        if i ~= 1
            
            prev = peaks(:, i - 1);
            
            min_peak = min(curr(2), prev(2));
            min_freq = min(start_freq, end_freq);
            
            d = min_peak - min_freq;
            
            a = min_freq + d * 0.3 + d * 0.3 * rand();
            
            time = (curr(1) + prev(1)) / 2;
            nodes = [ nodes, [ time; a] ];

        end
        
        nodes = [ nodes, curr ];
        
    end

    nodes = [ nodes, end_point ];
        
    t1 = t*duration + start_time;
    %s = spline(nodes(1, :), [0, nodes(2, :), 0], t1);        
    s = interp1(nodes(1, :), nodes(2, :), t1, 'pchip');        
    points = [t1; s];        
    
    vibrato = P.vibrato_amp * sin(points(1, :) * 2 * pi / vibrato_period);    
    
    points(2, :) = points(2, :) + vibrato;
        
    voc = points;
      

    
    
    
