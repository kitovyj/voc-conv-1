function prepare_breaks()
 
    path = '.\breaks\';
    mkdir(path);
    
    data_path = '.\breaks_temp\';
    
    count = 0;
    total_classes = [0, 0];
        
    class_folders = {};
    breaks_folders = {'a', 'b'};
    
    for i = 1:numel(total_classes)        
        class_folder = strcat(data_path, sprintf('%d\\', i));
        mkdir(class_folder);
        class_folders{i} = class_folder;
    end
    
    for i = 1:numel(total_classes)
        
        dest_folder = class_folders{i};        
        src_folder = strcat(path, breaks_folders{i});
        src_folder = strcat(src_folder, '\\');
            
        source_files = strcat(src_folder, '*.png');

        files = dir(source_files);
        
        test_amount = 250;    

        for file = files'
                        
            class_id = i - 1;
        
            source_file_png = strcat(src_folder, file.name);
            
            [~, name, ~] = fileparts(file.name);
            
            source_file_png_raw = strcat('.\data_unbalanced_new\', name, 'r', '.png');
            source_file_mat = strcat('.\data_unbalanced_new\', name, '.mat');
            source_file_csv = strcat('.\data_unbalanced_new\', name, '.csv');

            class_count = total_classes(class_id + 1);
            
            if class_count < test_amount
                file_name = strcat(dest_folder, sprintf('test%09d', class_count)); 
            else    
                file_name = strcat(dest_folder, sprintf('data%09d', class_count - test_amount)); 
            end
            
            file_name_raw = strcat(file_name, 'r');             
            png_raw_file = strcat(file_name_raw, '.png');
            
            png_file = strcat(file_name, '.png');
            csv_file = strcat(file_name, '.csv');
            mat_file = strcat(file_name, '.mat');
            
            copyfile(source_file_png, png_file);
            copyfile(source_file_png_raw, png_raw_file);
            copyfile(source_file_mat, mat_file);
                        
            labels = csvread(source_file_csv);
            
            labels(1) = class_id;
            
            csvwrite(csv_file, labels);
        
            count = count + 1;
            total_classes(class_id + 1) = total_classes(class_id + 1) + 1;

        end
                
    end    
    
    for i = 1:numel(total_classes)
        fprintf('class %d: %d\n', int32(i), int32(total_classes(i)));        
    end
    
    % [4651, 3636]
    % count = equalize([4651, 3636], 1, '.\breaks_data\', '.\breaks_temp\')
    % count = 8802
    % permute_vocalizations(8802, '.\breaks_data\');