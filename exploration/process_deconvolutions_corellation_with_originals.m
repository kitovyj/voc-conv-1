function process_deconvolutions_correlation_with_originals()


    sdata = csvread('features_fileid_sex_sexpredictiondnn_duration_avgfreq_avgvol_direction_peaks_breaks_broadband_tremolo_complex.csv');    

    data_path = './deconv/';
    originals_path = './originals/';
    
    dirs = dir(data_path);    

    % process the corellation of neurons in the individual layers 
    
    
    layer_folders = { 'relu', 'relu_1', 'relu_2', 'relu_3', 'relu_4', 'relu_5', 'fc_0', 'fc_1', 'fc_2' };
        
    image_size = 100;
     
    male = [];
    female = [];
    
    count = 1;
    ids = [];
    
    for folder = dirs'

        if(folder.name == "." || folder.name == "..")
           continue
        end

        [~, fname, ~] = fileparts(folder.name);
        id = str2double(fname);
        
        ofn = strcat(originals_path, sprintf('data%09d.png', id)); 
        original = imread(ofn);
        original = double(reshape(original, 1, []));                                              
        
        cc = [id];
             
        for n = 1:numel(layer_folders)
            
            %disp(layer_folders{n});
            
            image_fn = strcat(data_path, folder.name, '/', layer_folders{n}, '/deconvolution/grid_image.png');                
            image = imread(image_fn);
             
            % first six layers are convolution layers having 256 neurons
            % each
            if(n < 7)                
                % convolutional layers are arranged in 16x16 grids                
                grid_size = 16;                                                                
            else                
                grid_size = 11;
            end
                        
            data = [];
            
            % disp(n);
            % disp(grid_size);
            
            for i = 1:grid_size
                for j = 1:grid_size
                                
                    % omit the last image for fully connected laters
                    if(n >= 7 && (j == grid_size && i == grid_size))
                        continue
                    end
                    
                    % take a 1-pixel width separator line into accaunt
                    y = (j - 1) * (image_size + 1) + 1;
                    x = (i - 1) * (image_size + 1) + 1;
                        
                    %disp(x);
                    %disp(y);
                    
                    
                    sub_image = image(y:(y + image_size - 1), x:(x + image_size - 1), :);                        
                    % flatten
                    d = double(reshape(sub_image, 1, [])); 
                    
                    c = corrcoef(d, original);
                      
                    data = [data; c(2, 1)];
                        
                end
            end                                

            c = nanmean(data);
                        
            cc = [cc c];
                                    
        end
        
        disp(count)
        disp(id)
        count = count + 1;
        disp(cc)
        
        row = sdata(sdata(:, 1) == id, :);
        sex = row(3);
        if(sex == 0)            
           female = [female; cc];
        else            
           male = [male; cc];            
        end
        
    end

    disp(mean(female))
    disp(std(female) / sqrt(length(female)))    
    disp(mean(male))
    disp(std(male) / sqrt(length(male)))   
        
    save('deconv_original_correlations_female', 'female'); 
    save('deconv_original_correlations_male', 'male'); 
