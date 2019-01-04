function test_corellations()

    sdata = csvread('features_fileid_sex_sexpredictiondnn_duration_avgfreq_avgvol_direction_peaks_breaks_broadband_tremolo_complex.csv');    

    data_path = './conv_activations/';
    originals_path = './originals/';
    
    dirs = dir(data_path);
    dir_indices = randperm(numel(dirs), numel(dirs));      
    
    female = [];    
    male = [];
        
    for di = dir_indices
        
        d = dirs(di);
        
        if(d.name == "." || d.name == "..")
            continue
        end
        
        [~, dname, ~] = fileparts(d.name);

        id = str2double(dname);

        ofn = strcat(originals_path, sprintf('data%09d.png', id)); 
        original = imread(ofn);
        original = double(reshape(original, 1, []));          
        
        row = sdata(sdata(:, 1) == id, :);        
        sex = row(3);

        if sex == 0
            if size(female, 1) >= 1000
                continue
            end
        else
            if size(male, 1) >= 1000
                continue
            end
        end
        % disp(sex);
        
%        for l = 1:6
        
        layer = 6;

        a = original;                    
 %       end
 
        if sex == 0            
            female = [female; a];            
        else
            male = [male; a];
        end    
        
    end
    
    d = corr(female');
    I = tril(logical(ones(size(d))),-1);
    ltd = d(I);
    nanmean(ltd)
 
    d = corr(male');
    I = tril(logical(ones(size(d))),-1);
    ltd = d(I);
    nanmean(ltd) 
   
    d = corr(female', male');
    I = tril(logical(ones(size(d))),-1);
    ltd = d(I);
    nanmean(ltd) 
    