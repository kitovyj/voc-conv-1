function collect_vocalizations()

 
    path = 'c:\DNN\hammerschmidt\';
    data_path = '.\data\';
    
    count = 0;
    
    source_files = strcat(path, '*.wav');

    files = dir(source_files);

    for file = files'
        
        male = 1;
        if strfind(file.name, 'Rfem') == 1
            male = 0;
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

            file_name = strcat(data_path, sprintf('data%09d', count)); 
            png_file = strcat(file_name, '.png');
            csv_file = strcat(file_name, '.csv');
    
            imwrite(im, png_file);
        
            labels = [male];
            csvwrite(csv_file, labels);
        
            count = count + 1;

        end
    end    
    
    % permute
            
    disp('permuting...');
    
    temp_png_file = strcat(data_path, 'temp.png'); 
    temp_csv_file = strcat(data_path, 'temp.csv'); 
    
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

        file_name_b = strcat(data_path, sprintf('data%09d', b)); 
        png_file_b = strcat(file_name_b, '.png');
        csv_file_b = strcat(file_name_b, '.csv');
        
        movefile(png_file_a, temp_png_file);
        movefile(csv_file_a, temp_csv_file);
        
        movefile(png_file_b, png_file_a);
        movefile(csv_file_b, csv_file_a);
        
        movefile(temp_png_file, png_file_b);
        movefile(temp_csv_file, csv_file_b);
                
    end
    
    fprintf('\n');
    
    
    