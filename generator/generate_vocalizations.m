function generate_vocalizations(varargin)

    storage_a = vocalizations_storage('./data/a/');
    storage_b = vocalizations_storage('./data/b/');

    for i=1:100000
        
        v = generate_vocalization('vibrato_amp', 400);        
        storage_a.store(v);
        
        v = generate_vocalization('vibrato_amp', 0);        
        storage_b.store(v);

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
        
        
      

    
    
    
    