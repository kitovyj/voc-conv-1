function collect_mouse_ids()

    data_path = 'd:\dnn\voc-conv-1\classifier\data_unbalanced_new\';
    
    source_files = strcat(data_path, '*.mat');

    files = dir(source_files);
    
    for file = files'
                
        mat_name = file.name;
        
        [~, file_name, ~] = fileparts(mat_name);
        
        mat_file = strcat(data_path, mat_name);
        
        id_name = strcat(file_name, '.id');
        id_file = strcat(data_path, id_name);

        load(mat_file);
                
        % compute average frequency
        
        %spec = v.Spec{1};
        
        [~, file_name, ~] = fileparts(v.Filename);
      
        id = fopen(id_file, 'w');
        fprintf(id, file_name);
        fclose(id);
        
    end
    
