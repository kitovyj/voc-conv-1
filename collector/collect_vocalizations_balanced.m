function collect_vocalizations_balanced()
 
    path = 'd:\DNN\hammerschmidt\';
    
    data_path = '.\data_temp_1\';
    
    count = 0;
    total_classes = [0, 0];
        
    class_folders = {};
    
    for i = 1:numel(total_classes)        
        class_folder = strcat(data_path, sprintf('%d\\', i));
        mkdir(class_folder);
        class_folders{i} = class_folder;
    end

    source_files = strcat(path, '*.wav');

    files = dir(source_files);
        
    test_amount = 250;    

    for file = files'
                        
        class_id = 1;
        if strfind(file.name, 'Rfem') == 1
            class_id = 0;
        end
        
        source_file = strcat(path, file.name);
            
        vocs = VocCollector('DataSource', 'WAV', 'Filename', source_file, 'FRange', [10000,1500000]);

        total_vocs = numel(vocs);
        percent = int64(total_vocs / 100);
        
        for i=1:numel(vocs)
        
            if mod(i - 1, percent*10) == 0            
                fprintf('%d%%', int32(round((i - 1) * 100 / total_vocs)));
            elseif mod(i - 1, percent) == 0
                fprintf('.');
            end
                    
            freq_range = 233;
       
            im = zeros(freq_range, 100);

            sp = vocs(i).Spec{1};
            
            sp = flipud(sp);
            
            start_row = 1;
            start_col = 1;
            im(start_row: start_row + size(sp, 1) - 1, start_col:start_col + size(sp, 2) - 1) = sp;
        
            im = imresize(im, [100 100]);

            class_folder = class_folders{class_id + 1};
            
            class_count = total_classes(class_id + 1);

            if class_count < test_amount
                file_name = strcat(class_folder, sprintf('test%09d', class_count)); 
            else    
                file_name = strcat(class_folder, sprintf('data%09d', class_count - test_amount)); 
            end
                        
            file_name_raw = strcat(file_name, 'r');             
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
            
            avg_value = mean(mean(spec));
            
            labels = [class_id, v.Duration, avg_freq, avg_value];
            
            csvwrite(csv_file, labels);
        
            count = count + 1;
            total_classes(class_id + 1) = total_classes(class_id + 1) + 1;

        end
                
    end    
    
    for i = 1:numel(total_classes)
        fprintf('class %d: %d\n', int32(i), int32(total_classes(i)));        
    end

    % equalize, [6477, 4492]
    % new : [5760, 4361]
    
    count = equalize(total_classes, 0, '.\data\', '.\data_temp\');
    
    % permute
    
    fprintf('total objects to permute: %d\n', int32(count));        
        
    permute_vocalizations(count, '.\data\');
    
    