function generate_vocalizations(varargin)

    storage_a = vocalizations_storage('./data/a/');
    storage_b = vocalizations_storage('./data/b/');
    storage_c = vocalizations_storage('./data/c/');
    storage_d = vocalizations_storage('./data/d/');
    storage_e = vocalizations_storage('./data/e/');
    storage_f = vocalizations_storage('./data/f/');
    storage_g = vocalizations_storage('./data/g/');
    storage_h = vocalizations_storage('./data/h/');

    for i=1:20000
        
        %{
        v = generate_vocalization('vibrato_amp', 400);        
        storage_a.store(v);
        
        v = generate_vocalization('vibrato_amp', 0);        
        storage_b.store(v);
        %}

        v = generate_vocalization('vibrato_amp', 0, 'peaks_num', 1);
        storage_a.store(v);
        
        v = generate_vocalization('vibrato_amp', 400, 'peaks_num', 1);
        storage_b.store(v);

        v = generate_vocalization('vibrato_amp', 0, 'peaks_num', 2);
        storage_c.store(v);
        
        v = generate_vocalization('vibrato_amp', 400, 'peaks_num', 2);
        storage_d.store(v);
        
        v = generate_vocalization('vibrato_amp', 0, 'peaks_num', 3);
        storage_e.store(v);
        
        v = generate_vocalization('vibrato_amp', 400, 'peaks_num', 0);
        storage_f.store(v);

        v = generate_vocalization('vibrato_amp', 0, 'peaks_num', 0);
        storage_g.store(v);
        
        v = generate_vocalization('vibrato_amp', 400, 'peaks_num', 3);
        storage_h.store(v);

        %{
        hold on;
        cla;
        xlim([0 0.15])
        ylim([20000 130000])
        xlabel('s')
        ylabel('Hz')
        plot(v(1,:), v(2,:))
        hold off;
        %}
        
        %waitforbuttonpress;
        
    end
        
        
      

    
    
    
    