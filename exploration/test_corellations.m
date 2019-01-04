function test_corellations()


    sdata = csvread('features_fileid_sex_sexpredictiondnn_duration_avgfreq_avgvol_direction_peaks_breaks_broadband_tremolo_complex.csv');    

    data_path = './activations/';
    
    dirs = dir(data_path);
    

    for n = 0:2

        male = [];
        female = [];
    
        for d = dirs'
        
            if(d.name == "." || d.name == "..")
                continue
            end
                   
            lns = int2str(n);
            wfn = strcat(data_path, d.name, '/fc_', lns, '/activations/grid_activation.png');                
            w = imread(wfn);        
            w = double(reshape(w, 1, []));

            [~, dname, ~] = fileparts(d.name);

            id = str2double(dname);

            row = sdata(sdata(:, 1) == id, :);

            sex = row(3);

            if(sex == 0)            
                female = [female; w];            
            else            
                male = [male; w];            
            end
                        
        end
        
        lns = int2str(n + 1);
        wfn = strcat('female_', lns);                     
        save(wfn, 'female'); 

        wfn = strcat('male_', lns);                     
        save(wfn, 'male'); 
        
    end
    
    %{
    if(size(female, 1) > size(male, 1))
        
        female = female(1:size(male, 1), :);
        
    else

        male = male(1:size(female, 1), :);
        
    end
    %}
    

    % male
    %female
    
    %female
    
    %{
    s = size(female, 1);
    cc = corrcoef(female(1:s/2, :), female(s/2 + 1:end, :))
    
    s = size(male, 1);
    cc = corrcoef(male(1:s/2, :), male(s/2 + 1:end, :))
    
    
    cc = corrcoef(female, male)
    %}
    
    
    
    d = corr(male');
    %DUnique = tril(d,-1)
    I = tril(logical(ones(size(d))),-1);
    ltd = d(I);
    nanmean(ltd)
    
    
    
