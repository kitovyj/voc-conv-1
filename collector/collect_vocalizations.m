function collect_vocalizations()
 
    path = 'c:\DNN\hammerschmidt\';
    data_path = '.\data\';
    
    count = 0;
    total_classes = [0, 0];
    
    source_files = strcat(path, '*.wav');

    files = dir(source_files);

    for file = files'
        
        male = 1;
        if strfind(file.name, 'Rfem') == 1
            male = 0;
        end
        
        source_file = strcat(path, file.name);
            
        vocs = VocCollector('DataSource', 'WAV', 'Filename', source_file, 'FRange', [10000,1500000]);
        
        % [W, SR]= audioread('c:\DNN\hammerschmidt\Rfem_Afem01.wav');
        % RW = C_convertWAV2Controller('Data', W, 'SR', SR);
        % MultiViewer('Data', RW, 'DetectVocs', 1)
        

        for i=1:numel(vocs)
        
            freq_range = 233;
       
            im = zeros(freq_range, 100);

            sp = vocs(i).Spec{1};

            start_row = 1;
            start_col = 1;
            im(start_row: start_row + size(sp, 1) - 1, start_col:start_col + size(sp, 2) - 1) = sp;
        
            im = imresize(im, [100 100]);

            file_name = strcat(data_path, sprintf('data%09d', count)); 
            file_name_raw = strcat('r', file_name);             
            png_raw_file = strcat(file_name_raw, '.png');
            png_file = strcat(file_name, '.png');
            csv_file = strcat(file_name, '.csv');
            mat_file = strcat(file_name, '.mat');
            
            v = vocs(i);
            save(mat_file, 'v');
    
            imwrite(im, png_file);
            imwrite(sp, png_raw_file);
                    
            spec = v.Spec{1};
            height = size(spec, 1);
            coord = linspace(0, height - 1, height)';
            coord = repmat(coord, 1, size(spec, 2));        
            freq_sums = sum(spec, 1);
        
            weighted_coord = spec .* coord;
            weighted_sums = sum(weighted_coord, 1);
            avg_freq = mean(weighted_sums ./ freq_sums);
            
            labels = [male, v.Duration, avg_freq];
            csvwrite(csv_file, labels);
        
            count = count + 1;

        end
        
        if male == 0
            total_classes(2) = total_classes(2) + numel(vocs);
        else
            total_classes(1) = total_classes(1) + numel(vocs);
        end
        
    end    
    
    for i = 1:numel(total_classes)
        fprintf('class %d: %d\n', int32(i), int32(total_classes(i)));        
    end
    
    % permute
    
    permute_vocalizations(count)
            
    
    
    