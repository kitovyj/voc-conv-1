function k = equalize(total_classes)
 
    data_path = '.\data\';
            
    disp('equalizing');
    
    class_folders = [];
    
    for i = 1:numel(total_classes)        
        class_folder = strcat(data_path, sprintf('%d\\', i));
        class_folders{i} = class_folder;
    end    
    
    max_amount = max(total_classes);
    
    k = 0;
    
    for i = 1:numel(total_classes)

        class_folder = class_folders{i};
        
        for j = 1:total_classes(i)

            file_name = strcat(class_folder, sprintf('data%09d', j - 1)); 
            png_file = strcat(file_name, '.png');
            csv_file = strcat(file_name, '.csv');
            mat_file = strcat(file_name, '.mat');

            file_name = strcat(data_path, sprintf('data%09d', k)); 
            dest_png_file = strcat(file_name, '.png');
            dest_csv_file = strcat(file_name, '.csv');
            dest_mat_file = strcat(file_name, '.mat');
            
            copyfile(png_file, dest_png_file);
            copyfile(csv_file, dest_csv_file);
            copyfile(mat_file, dest_mat_file);    
            
            k = k + 1;  
        end
        
        if total_classes(i) < max_amount

            fprintf('class %d: cloning extra %d examples\n', int32(i), int32(max_amount - total_classes(i)));        
            
            for m = 1:(max_amount - total_classes(i))
            
                r = int64(round(rand()*(total_classes(i) - 1)));
                
                file_name = strcat(class_folder, sprintf('data%09d', r)); 
                png_file = strcat(file_name, '.png');
                csv_file = strcat(file_name, '.csv');
                mat_file = strcat(file_name, '.mat');

                file_name = strcat(data_path, sprintf('data%09d', k)); 
                dest_png_file = strcat(file_name, '.png');
                dest_csv_file = strcat(file_name, '.csv');
                dest_mat_file = strcat(file_name, '.mat');

                copyfile(png_file, dest_png_file);
                copyfile(csv_file, dest_csv_file);
                copyfile(mat_file, dest_mat_file);    
                
                k = k + 1;  
                
            end
        end
        
    end            
    
    
    