function permute_vocalizations(count, data_path)

    % permute
            
    disp('permuting...');
    
    temp_png_file = strcat(data_path, 'temp.png'); 
    temp_png_raw_file = strcat(data_path, 'tempr.png'); 
    temp_csv_file = strcat(data_path, 'temp.csv'); 
    temp_mat_file = strcat(data_path, 'temp.mat'); 
    
    percent = int64(count / 100);
    
    for i = 1:count
        
        if mod(i - 1, percent*10) == 0            
            fprintf('%d%%', int32(round((i - 1) * 100 / count)));
        elseif mod(i - 1, percent) == 0
            fprintf('.');
        end
        
        a = int64(round(rand()*(count - 1)));
        b = a;
        
        while b == a
            b = int64(round(rand()*(count - 1)));
        end
        
        file_name_a = strcat(data_path, sprintf('data%09d', a)); 

        file_name_raw_a = strcat(file_name_a, 'r');             
        png_raw_file_a = strcat(file_name_raw_a, '.png');           
        
        png_file_a = strcat(file_name_a, '.png');
        csv_file_a = strcat(file_name_a, '.csv');
        mat_file_a = strcat(file_name_a, '.mat');

        file_name_b = strcat(data_path, sprintf('data%09d', b)); 

        file_name_raw_b = strcat(file_name_b, 'r');             
        png_raw_file_b = strcat(file_name_raw_b, '.png');           
        
        png_file_b = strcat(file_name_b, '.png');
        csv_file_b = strcat(file_name_b, '.csv');
        mat_file_b = strcat(file_name_b, '.mat');
        
        movefile(png_file_a, temp_png_file);
        movefile(png_raw_file_a, temp_png_raw_file);
        movefile(csv_file_a, temp_csv_file);
        movefile(mat_file_a, temp_mat_file);
        
        movefile(png_file_b, png_file_a);
        movefile(png_raw_file_b, png_raw_file_a);
        movefile(csv_file_b, csv_file_a);
        movefile(mat_file_b, mat_file_a);
        
        movefile(temp_png_file, png_file_b);
        movefile(temp_png_raw_file, png_raw_file_b);
        movefile(temp_csv_file, csv_file_b);
        movefile(temp_mat_file, mat_file_b);
                
    end
    
    fprintf('\n');
    
    
