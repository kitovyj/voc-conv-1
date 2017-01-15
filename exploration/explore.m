function explore()

    data_path = '..\collector\data\';
    
    source_files = strcat(data_path, '*.mat');

    files = dir(source_files);

    duration_sum = [0, 0];
    total_classes = [0, 0];    
    
    count = numel(files);
    
    percent = int64(count / 100);
        
    i = 0;
    
    for file = files'

        if mod(i, percent*10) == 0            
            fprintf('%d%%', int32(round(i * 100 / count)));
        elseif mod(i, percent) == 0
            fprintf('.');
        end
        
        i = i + 1;
        
        mat_name = file.name;
        
        [~, file_name, ~] = fileparts(mat_name);
        
        csv_name = strcat(file_name, '.csv');
                        
        mat_file = strcat(data_path, mat_name);
        csv_file = strcat(data_path, csv_name);
        
        load(mat_file);
        labels = csvread(csv_file);
        
        class_num = int32(labels(1)) + 1; 
        duration_sum(class_num) = duration_sum(class_num) + v.Duration;
        total_classes(class_num) = total_classes(class_num) + 1;
        
        %v.Duration
        %labels
                
    end
    
    fprintf('\n');
    
    avg_duration = duration_sum ./ total_classes;
    
    for i = 1:numel(avg_duration)
        fprintf('average duration: %f\n', avg_duration(i));        
    end
    
