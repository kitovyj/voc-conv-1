function analyze_deconvolutions()
        
    %format long;
    format;

    load('deconv_original_correlations_female.mat', 'female'); 
    load('deconv_original_correlations_male.mat', 'male'); 

    disp(mean(female(1:end, 2:end)))
    disp(std(female(1:end, 2:end)) / sqrt(length(female)))    
    disp(mean(male(1:end, 2:end)))
    disp(std(male(1:end, 2:end)) / sqrt(length(male)))   
    
    sdata = csvread('features_fileid_sex_sexpredictiondnn_duration_avgfreq_avgvol_direction_peaks_breaks_broadband_tremolo_complex.csv');    

    abnormal_f = female(abs(female(:, 8)) < 0.1, :);
    disp(length(abnormal_f));
    disp(length(female));
    
    %disp(abnormal_f);
    
    abnormal_m = male(abs(male(:, 8)) < 0.1, :);
    disp(length(abnormal_m));    
    disp(length(male));
           
    %disp(abnormal_m);
    
    %disp(abnormal_m(:, 1));

    a_sdata_f = sdata(ismember(sdata(:, 1), abnormal_f(:, 1)), :);

    %disp(a_sdata_f);
    %disp(abnormal_f(:, 1));
    %disp(ismember(sdata(:, 1), abnormal_f(:, 1)));
    
    correct_abnormal_f = a_sdata_f(a_sdata_f(:, 3) == a_sdata_f(:, 4));
    disp(size(correct_abnormal_f, 1) / size(abnormal_f, 1));
        
    a_sdata_m = sdata(ismember(sdata(:, 1), abnormal_m(:, 1)), :);
    correct_abnormal_m = a_sdata_m(a_sdata_m(:, 3) == a_sdata_m(:, 4));
    disp(size(correct_abnormal_m, 1) / size(abnormal_m, 1));

    
    load('deconv_inner_correlations_female.mat', 'female'); 
    load('deconv_inner_correlations_male.mat', 'male'); 

    format;
    
    disp(mean(female(1:end, 2:end)))
    disp(std(female(1:end, 2:end)) / sqrt(length(female)))    
    disp(mean(male(1:end, 2:end)))
    disp(std(male(1:end, 2:end)) / sqrt(length(male)))   
