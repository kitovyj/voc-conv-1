function peaks = generate_peaks(x1, y1, x2, y2, peak_min_amp, peak_max_amp, peaks_num)
        
    if peaks_num == 0
       
        peaks = [];
        return;
        
    end


    d = x2 - x1;
    m = 0.2 * d;
    xs = x1 + m;
    xe = x2 - m;
    d1 = xe - xs;
              
    peak_time = xs + d1 * rand();    
    peak_freq = peak_min_amp + rand() * (peak_max_amp - peak_min_amp);            
    
    p = [peak_time; peak_freq];        
    
    peaks_num = peaks_num - 1;
    
    if peaks_num == 0
       
        peaks = p;
        
    else
                
        if peak_time - xs > xe - peak_time
            
            nx1 = xs;
            nx2 = peak_time;
            ny1 = y1;
            ny2 = peak_freq;
            before = 1;
                   
        else

            nx1 = peak_time;
            nx2 = x2;
            ny1 = peak_freq;
            ny2 = y2;                        
            before = 0;

        end
                    
        inner_peaks = generate_peaks(nx1, ny1, nx2, ny2, peak_min_amp, peak_max_amp, peaks_num);
        
        if before == 1
            
            peaks = [inner_peaks, p];
            
        else
            
            peaks = [p, inner_peaks];
            
        end
        
    end
        
      

    
    
    
