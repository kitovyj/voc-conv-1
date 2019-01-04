function [fr, mr, cr] = test_corellations_conv(layer, feature, display)

    if display == 1
        fprintf("processing layer %d\n", layer);
    end
    
    sdata = csvread('features_fileid_sex_sexpredictiondnn_duration_avgfreq_avgvol_direction_peaks_breaks_broadband_tremolo_complex.csv');    

    data_path = './conv_activations/';
    
    dirs = dir(data_path);
    rng(1);
    dir_indices = randperm(numel(dirs), numel(dirs));      

    afn = strcat(data_path, '0', '/', int2str(layer), '.mat');
    load(afn, 'activations')
    
    if feature == 0    
        activations = reshape(activations(:, :, :, :), 1, []);
    else
        activations = reshape(activations(:, :, :, feature), 1, []);
    end
    
    %max_samples_male = size(sdata(sdata(:, 3) == 1, :), 1);
    %max_samples_female = size(sdata(sdata(:, 3) == 0, :), 1);

    max_samples_male = 3000;
    max_samples_female = 3000;
    
    female = zeros(numel(activations), max_samples_female);    
    male = zeros(numel(activations), max_samples_male);    
    
    female_index = 1;
    male_index = 1;
    
    count = 1;
    
    for di = dir_indices
        
        d = dirs(di);
        
        if(d.name == "." || d.name == "..")
            continue
        end
        
        count = count + 1;
        percents_done = (count * 100) / numel(dir_indices);
        if percents_done > 0 && mod(percents_done, 5) == 0
            disp(percents_done)
        end
                   
        [~, dname, ~] = fileparts(d.name);

        id = str2double(dname);

        row = sdata(sdata(:, 1) == id, :);        
        sex = row(3);

        if sex == 0
            if female_index > max_samples_female
                continue
            end
        else
            if male_index > max_samples_male
                continue
            end
        end
        
        % disp(sex);
        
%        for l = 1:6

        afn = strcat(data_path, d.name, '/', int2str(layer), '.mat');
        load(afn, 'activations')
        
        %imagesc(squeeze(activations(1, :, :, 1)));

        if feature == 0    
            a = reshape(activations(:, :, :, :), 1, []);
        else
            a = reshape(activations(:, :, :, feature), 1, []);
        end
                          
 %       end
 
        if sex == 0            
            female(:, female_index) = a;            
            female_index = female_index + 1;
        else
            male(:, male_index) = a;
            male_index = male_index + 1;
        end    
        
    end
    
    if female_index - 1 ~= size(female, 2)
        female = female(:,1:(female_index - 1));
    end
    
    if male_index - 1 ~= size(male, 2)
        male = male(:,1:(male_index - 1));
    end
        
    d = corr(female);
    I = tril(logical(ones(size(d))),-1);
    ltd = d(I);
    fr = ltd;
    
    if display == 1
        disp(nanmean(ltd));
        disp(nanstd(ltd) / sqrt(nnz(~isnan(ltd))));    
    end
        
    d = corr(male);
    I = tril(logical(ones(size(d))),-1);
    ltd = d(I);
    mr = ltd;

    if display == 1
        disp(nanmean(ltd));
        disp(nanstd(ltd) / sqrt(nnz(~isnan(ltd))));    
    end
    
    d = corr(female, male);
    I = tril(logical(ones(size(d))),-1);
    ltd = d(I);
    cr = ltd;
    
    if display == 1
        disp(nanmean(ltd));
        disp(nanstd(ltd) / sqrt(nnz(~isnan(ltd))));   
    end    
    
    clearvars -except fr mr cr;