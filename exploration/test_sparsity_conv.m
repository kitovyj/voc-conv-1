function test_corellations()

    sdata = csvread('features_fileid_sex_sexpredictiondnn_duration_avgfreq_avgvol_direction_peaks_breaks_broadband_tremolo_complex.csv');    

    data_path = './conv_activations/';
    
    dirs = dir(data_path);
            
    female = [];    
    male = [];
    
    for d = dirs'
        
        if(d.name == "." || d.name == "..")
            continue
        end
        
        [~, dname, ~] = fileparts(d.name);

        id = str2double(dname);

        row = sdata(sdata(:, 1) == id, :);        
        sex = row(3);
        
        % disp(sex);

        layer_sparsity = [0, 0, 0, 0, 0, 0];
        
        for l = 1:6
            
            afn = strcat(data_path, d.name, '/', int2str(l), '.mat');
            load(afn, 'activations')
            a = reshape(activations, 1, []);
            nz = nnz(a) / numel(a);
            
            layer_sparsity(l) = nz;
        
        end

        if sex == 0
            female = [female; layer_sparsity];
        else
            male = [male; layer_sparsity];            
        end
                            
    end
    
    disp(mean(female))
    disp(std(female) / sqrt(length(female)))    
    disp(mean(male))
    disp(std(male) / sqrt(length(male)))  
    