function explore()

    data_path = '..\collector\data\';
        
    source_files = strcat(data_path, '*.mat');

    files = dir(source_files);

    count = numel(files);

    duration_sum = [0, 0];
    total_classes = [0, 0];    
    
    durations = { []; []; };
    frequencies = { []; []; };
    loudnesses = { []; []; };
        
    percent = int64(count / 100);
        
    i = 0;
    
    freq_range = 233;
    time_range = 100;
    
    total_pixels = freq_range * time_range;
    total_pixels_100 = 100 * 100;

    raw = zeros(total_pixels, count);
    raw_100 = zeros(total_pixels_100, count);
    responses = zeros(count, 1);
    
    for file = files'
        
        %disp(file);

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
        
        avg_value = mean(mean(spec));

        loudnesses{class_num} = [loudnesses{class_num} avg_value]; 
               
        im = zeros(freq_range, time_range);

        sp = spec(:, 1:min(size(spec, 2), time_range));
            
        start_row = 1;
        start_col = 1;
        im(start_row: start_row + size(sp, 1) - 1, start_col:start_col + size(sp, 2) - 1) = sp;
        
        raw(1:end, i) = im(:);
        
        im = imresize(im, [100 100]);

        raw_100(1:end, i) = im(:);
        
        responses(i) = class_num - 1; 
        
    end
    
    save('statistics.mat', 'total_classes', 'frequencies', 'durations', 'loudnesses', 'raw', 'responses', 'raw_100');
    
    fprintf('\n');

    analyze();
