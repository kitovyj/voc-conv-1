function average_all()

    total_classes = [5760, 4361];
    
    data_temp_path = '.\data_temp\';
        
    class_folders = [];
    images = [];
    figures = [];
    f_axes = [];
    
    images{1} = zeros(100, 100);
    images{2} = zeros(100, 100);
    
    figures{1} = figure();
    figures{2} = figure();
    f_axes{1} = axes(figures{1});
    f_axes{2} = axes(figures{2});
    
    
    for i = 1:numel(total_classes)        
        class_folder = strcat(data_temp_path, sprintf('%d\\', i));
        class_folders{i} = class_folder;
    end        
    
    test_amount = 250;
    
    
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

            png_file = strcat(file_name, '.png');
            
            im = imread(png_file);
            
            images{i} = images{i} + double(im);
            
            imagesc(images{i} / j, 'parent', f_axes{i});
            refresh(figures{i});
            drawnow;
                        
            
        end
        
    end            
    
    
end    