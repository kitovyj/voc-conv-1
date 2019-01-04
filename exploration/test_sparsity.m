function test_corellations()



    data_path = './activations/';
    
    dirs = dir(data_path);
    
    fc0_sparsity = 0;
    fc1_sparsity = 0;
    fc2_sparsity = 0;
    
    count = 0;
    
    for d = dirs'
        
        if(d.name == "." || d.name == "..")
            continue
        end
        
        wfn = strcat(data_path, d.name, '/fc_0/activations/grid_activation.png');                
        w = imread(wfn);
        w = reshape(w, 1, []);
        nz = nnz(w);        
        fc0_sparsity = fc0_sparsity + nz;
        
        wfn = strcat(data_path, d.name, '/fc_1/activations/grid_activation.png');                
        w = imread(wfn);
        w = reshape(w, 1, []);
        nz = nnz(w);        
        fc1_sparsity = fc1_sparsity + nz;
    
        wfn = strcat(data_path, d.name, '/fc_2/activations/grid_activation.png');                
        w = imread(wfn);
        w = reshape(w, 1, []);
        nz = nnz(w);        
        fc2_sparsity = fc2_sparsity + nz;
              
        count = count + 1;
        
    end
    
    fc0_sparsity = fc0_sparsity / count
    fc1_sparsity = fc1_sparsity / count
    fc2_sparsity = fc2_sparsity / count
