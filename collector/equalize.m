function k = equalize(total_classes, do_equalize, data_path, data_temp_path)
 
    test_amount = 250;
            
    disp('equalizing');
    
    mkdir(data_path);
    
    class_folders = [];
    
    for i = 1:numel(total_classes)        
        class_folder = strcat(data_temp_path, sprintf('%d\\', i));
        class_folders{i} = class_folder;
    end    
    
    max_amount = max(total_classes);
    
    k = 0;
    test_k = 0;
    
    
    for i = 1:numel(total_classes)

        class_folder = class_folders{i};
        
        disp(class_folder)
    
        percent = int64(total_classes(i) / 100);
                
        for j = 1:total_classes(i)

            if mod(j - 1, percent*10) == 0            
                fprintf('%d%%', int32(round((j - 1) * 100 / total_classes(i))));
            elseif mod(j - 1, percent) == 0
                fprintf('.');
            end
            
            if j > test_amount            
                file_name = strcat(class_folder, sprintf('data%09d', j - 1 - test_amount));
            else
                file_name = strcat(class_folder, sprintf('test%09d', j - 1));                 
            end

            file_name_raw = strcat(file_name, 'r');             

            png_raw_file = strcat(file_name_raw, '.png');           
            png_file = strcat(file_name, '.png');
            csv_file = strcat(file_name, '.csv');
            mat_file = strcat(file_name, '.mat');
            
            if j > test_amount
                file_name = strcat(data_path, sprintf('data%09d', k)); 
                k = k + 1;
            else
                file_name = strcat(data_path, sprintf('test%09d', test_k)); 
                test_k = test_k + 1;
            end
            
            file_name_raw = strcat(file_name, 'r');             
            dest_png_raw_file = strcat(file_name_raw, '.png');           
           
            dest_png_file = strcat(file_name, '.png');
            dest_csv_file = strcat(file_name, '.csv');
            dest_mat_file = strcat(file_name, '.mat');
            
            copyfile(png_raw_file, dest_png_raw_file);
            copyfile(png_file, dest_png_file);
            copyfile(csv_file, dest_csv_file);
            copyfile(mat_file, dest_mat_file);
            
        end
        
        fprintf('\n');
                
        if (total_classes(i) < max_amount) && do_equalize
            
            extra = int64(max_amount - total_classes(i));

            fprintf('class %d: cloning extra %d examples\n', int32(i), extra);        

            percent = int64(extra / 100);
             
            for m = 1:extra

                if mod(int64(m - 1), percent*10) == 0            
                    fprintf('%d%%', int32(round((m - 1) * 100 / extra)));
                elseif mod(int64(m - 1), percent) == 0
                    fprintf('.');
                end                
                
                r = int64(round(rand()*(total_classes(i) - 1 - test_amount)));
                
                file_name = strcat(class_folder, sprintf('data%09d', r)); 

                
                png_file = strcat(file_name, '.png');
                csv_file = strcat(file_name, '.csv');
                mat_file = strcat(file_name, '.mat');

                file_name_raw = strcat(file_name, 'r');             
                png_raw_file = strcat(file_name_raw, '.png');           
                
                file_name = strcat(data_path, sprintf('data%09d', k)); 
                
                file_name_raw = strcat(file_name, 'r');             
                dest_png_raw_file = strcat(file_name_raw, '.png');           
                
                dest_png_file = strcat(file_name, '.png');
                dest_csv_file = strcat(file_name, '.csv');
                dest_mat_file = strcat(file_name, '.mat');

                copyfile(png_file, dest_png_file);
                copyfile(png_raw_file, dest_png_raw_file);
                copyfile(csv_file, dest_csv_file);
                copyfile(mat_file, dest_mat_file);
                
                k = k + 1;  
                
            end
            
            fprintf('\n');
    
        end
        
    end            
    
    
    