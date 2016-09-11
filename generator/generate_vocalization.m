function voc = generate_vocalizations(varargin)
        
    P = parsePairs(varargin);
    checkField(P, 'vibrato_amp', 1000);

    % all frequencies are in Hz
    freq_min = 40000; 
    freq_deviation = 20000;
    peak_deviation = 15000;
    
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
        
    peak_time = start_time + duration * rand();    
    peak_freq = max(start_freq, end_freq) + 2*rand() * peak_deviation;    
    
    p1 = [ start_time; start_freq ];
    p2 = [ peak_time; peak_freq ];
    p3 = [ start_time + duration; end_freq ];
    
    t = linspace(0, 1, 1000);
    
    points = kron((1 - t).^2, p1) + kron(2*(1 - t).*t, p2) + kron(t.^2, p3);
    
    vibrato = P.vibrato_amp * sin(points(1, :) * 2 * pi / vibrato_period);    
    
    points(2, :) = points(2, :) + vibrato;
        
    voc = points;
      

    
    
    
