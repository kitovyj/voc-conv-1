function explore()

    data_path = '..\collector\data\';
    
    source_files = strcat(data_path, '*.mat');

    files = dir(source_files);

    count = numel(files);

    duration_sum = [0, 0];
    total_classes = [0, 0];    
    
    durations = { []; []; };
    frequencies = { []; []; };
        
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

        durations{class_num} = [durations{class_num} v.Duration];
        %duration_sum(class_num) = duration_sum(class_num) + v.Duration;
        total_classes(class_num) = total_classes(class_num) + 1;
        
        % compute average frequency
        
        spec = v.Spec{1};
        height = size(spec, 1);
        coord = linspace(0, height - 1, height)';
        coord = repmat(coord, 1, size(spec, 2));
        
        freq_sums = sum(spec, 1);
        
        weighted_coord = spec .* coord;
        weighted_sums = sum(weighted_coord, 1);
        avg_freq = mean(weighted_sums ./ freq_sums);
        
        frequencies{class_num} = [frequencies{class_num} avg_freq];        

    end
    
    save('statistics.mat', 'total_classes', 'frequencies', 'durations');
    
    fprintf('\n');

    analyze();
