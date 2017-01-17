function collect_vocalizations_balanced()
 
    path = 'c:\DNN\hammerschmidt\';
    data_path = '.\data\';
    
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

    for file = files'
        
        class_id = 1;
        if strfind(file.name, 'Rfem') == 1
            class_id = 0;
        end
        
        source_file = strcat(path, file.name);
            
        vocs = VocCollector('DataSource', 'WAV', 'Filename', source_file, 'FRange', [10000,1500000]);

        for i=1:numel(vocs)
        
            freq_range = 233;
       
            im = zeros(freq_range, 100);

            sp = vocs(i).Spec{1};

            start_row = 1;
            start_col = 1;
            im(start_row: start_row + size(sp, 1) - 1, start_col:start_col + size(sp, 2) - 1) = sp;
        
            im = imresize(im, [100 100]);

            class_folder = class_folders{class_id + 1};
            
            file_name = strcat(class_folder, sprintf('data%09d', total_classes(class_id + 1))); 
            png_file = strcat(file_name, '.png');
            csv_file = strcat(file_name, '.csv');
            mat_file = strcat(file_name, '.mat');
            
            v = vocs(i);
            save(mat_file, 'v');
    
            imwrite(im, png_file);
        
            labels = [class_id];
            csvwrite(csv_file, labels);
        
            count = count + 1;
            total_classes(class_id + 1) = total_classes(class_id + 1) + 1;

        end
                
    end    
    
    for i = 1:numel(total_classes)
        fprintf('class %d: %d\n', int32(i), int32(total_classes(i)));        
    end

    % equalize
    
    count = equalize(total_classes);
    
    % permute
    
    permute_vocalizations(count);
    
    