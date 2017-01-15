function permute_vocalizations(count)

    % permute
            
    disp('permuting...');
    
    data_path = '.\data\';    
    
    temp_png_file = strcat(data_path, 'temp.png'); 
    temp_csv_file = strcat(data_path, 'temp.csv'); 
    temp_mat_file = strcat(data_path, 'temp.mat'); 
    
    percent = int64(count / 100);
    
    for i = 1:count
        
        if mod(i, percent*10) == 0            
            fprintf('%d%%', int32(round(i * 100 / count)));
        elseif mod(i, percent) == 0
            fprintf('.');
        end
        
        a = int64(round(rand()*(count - 1)));
        b = a;
        
        while b == a
            b = int64(round(rand()*(count - 1)));
        end
        
        file_name_a = strcat(data_path, sprintf('data%09d', a)); 
        png_file_a = strcat(file_name_a, '.png');
        csv_file_a = strcat(file_name_a, '.csv');
        mat_file_a = strcat(file_name_a, '.mat');

        file_name_b = strcat(data_path, sprintf('data%09d', b)); 
        png_file_b = strcat(file_name_b, '.png');
        csv_file_b = strcat(file_name_b, '.csv');
        mat_file_b = strcat(file_name_b, '.mat');
        
        movefile(png_file_a, temp_png_file);
        movefile(csv_file_a, temp_csv_file);
        movefile(mat_file_a, temp_mat_file);
        
        movefile(png_file_b, png_file_a);
        movefile(csv_file_b, csv_file_a);
        movefile(mat_file_b, mat_file_a);
        
        movefile(temp_png_file, png_file_b);
        movefile(temp_csv_file, csv_file_b);
        movefile(temp_mat_file, mat_file_b);
                
    end
    
    fprintf('\n');
    
    
