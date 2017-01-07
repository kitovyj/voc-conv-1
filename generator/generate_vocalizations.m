function generate_vocalizations(start_index)

    %rng(11);

    storage = vocalizations_storage('./data/', start_index);
    
    %{
    storage_a = vocalizations_storage('./data/a/');
    storage_b = vocalizations_storage('./data/b/');
    storage_c = vocalizations_storage('./data/c/');
    storage_d = vocalizations_storage('./data/d/');
    storage_e = vocalizations_storage('./data/e/');
    storage_f = vocalizations_storage('./data/f/');
    storage_g = vocalizations_storage('./data/g/');
    storage_h = vocalizations_storage('./data/h/');
    
    storages = [ storage_a, storage_b, storage_c, storage_d, storage_e, storage_f, storage_g, storage_h ];
    %}

    for i=1:200000
        
        
        vibrato = randi([0, 1], 1, 1);
        break_num = randi([0, 3], 1, 1);
        peak_num = randi([0, 3], 1, 1);
        
        vibrato_amp = 0;
        if vibrato == 1
            vibrato_amp = 400;
        end
        
        %if i == 26
        %    i = i + 1;
        %end
                
        v = generate_vocalization('vibrato_amp', vibrato_amp, 'peaks_num', peak_num);
        
        sparce_peaks = full(ind2vec(peak_num + 1, 4));
        sparce_breaks = full(ind2vec(break_num + 1, 4));
        labels = [sparce_peaks;sparce_breaks;vibrato]';
        
        storage.store(v, labels, break_num);   

        clear v;
        clear labels;
        
        %{
        index = peak_num * 2 + 1;
        if vibrato == 1
           index = index + 1;
        end
        
        st = storages(index);
        st.store(v, labels, with_a_break);
        %}
        
        %waitforbuttonpress;
        
    end
        
        
      

    
    
    
    