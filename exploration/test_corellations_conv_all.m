for i = 1:6
    female = [];
    male = [];
    cross = [];
    for f = 1:256
        tic;
        [fr, mr, cr] = test_corellations_conv(i, f, 0);
        toc;
        tic;
        
        if f == 1
            female = zeros(numel(fr), 256);
            male = zeros(numel(mr), 256);
            cross = zeros(numel(cr), 256);            
        end
        
        female(:, f) = fr;
        male(:, f) = mr;
        cross(:, f) = cr;
        
        toc;        
        disp(f);
    end 
    
    female = reshape(female, 1, []);
    male = reshape(male, 1, []);
    cross = reshape(cross, 1, []);
    
    ltd = female;
    disp(nanmean(ltd));
    disp(nanstd(ltd) / sqrt(nnz(~isnan(ltd))));    

    ltd = male;
    disp(nanmean(ltd));
    disp(nanstd(ltd) / sqrt(nnz(~isnan(ltd))));    
   
    ltd = cross;
    disp(nanmean(ltd));
    disp(nanstd(ltd) / sqrt(nnz(~isnan(ltd))));        
    
end