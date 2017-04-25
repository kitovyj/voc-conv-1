function visualize_features()

    data_path = '..\collector\data\';
    
    source_files = strcat(data_path, '*.mat');

    files = dir(source_files);

    count = numel(files);
        
    i = 0;
    
    features_fig = figure('Position', [100, 100, 300, 100]);    
    feature_list = uicontrol(features_fig, 'Style', 'listbox', 'Position', [0 0 300 100])
    
    image_fig = figure('Position', [100, 100, 800, 600]);
    hold on;
    
    for file = files'
        
        i = i + 1;
        
        mat_name = file.name;
        
        [~, file_name, ~] = fileparts(mat_name);
        
        csv_name = strcat(file_name, '.csv');
        csv_name = strcat('f', csv_name);
                        
        mat_file = strcat(data_path, mat_name);
        csv_file = strcat(data_path, csv_name);
        
        load(mat_file);
        features = csvread(csv_file);
        
        feature_strings = {};
        
        if features(1) == 1
            feature_strings{end + 1} = '1 peak';
        end

        if features(2) == 1
            feature_strings{end + 1} = '2 peaks';
        end

        if features(3) == 1
            feature_strings{end + 1} = '3 peaks';
        end

        if features(4) == 1
            feature_strings{end + 1} = '1 break';
        end

        if features(5) == 1
            feature_strings{end + 1} = '2 breaks';
        end

        if features(6) == 1
            feature_strings{end + 1} = '3 breaks';
        end

        if features(7) == 1
            feature_strings{end + 1} = 'vibrato';
        end
        
        if features(8) == 1
            feature_strings{end + 1} = 'ascending';
        end

        if features(9) == 1
            feature_strings{end + 1} = 'descending';
        end
        
        set(feature_list, 'string', feature_strings);
                
        % compute average frequency
        
        spec = v.Spec{1};
        
        freq_range = 233;
        
        im = zeros(freq_range, 100);

        sp = spec;

        start_row = 1;
        start_col = 1;
        im(start_row: start_row + size(sp, 1) - 1, start_col:start_col + size(sp, 2) - 1) = sp;
        im = imresize(im, [100 100]);
        
        
        clf(image_fig);
        figure(image_fig);
        %imshow(im, 'InitialMagnification', 'fit');
        
        time = linspace(0, 0.1, 100);
        
        imagesc(time, v.F, im);
        set(gca, 'ydir', 'normal');        
        waitforbuttonpress
        
    end
    
