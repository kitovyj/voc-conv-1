function analyze_raw_input()

    load('statistics.mat');

    % linear model
    
    data = raw_100;
    
    % data = data(:, 1:7000);
    % responses = responses(1:7000);

    % SVMModel = fitcsvm(X,Y,'KernelFunction','rbf','Standardize',true,'ClassNames', {'negClass', 'posClass'});
    
    %test_size = 500;
    test_size = 500;
    predictors_size = size(data, 2) - test_size;
    % predictors_matrix = [ data(:, 1:predictors_size); ones(1, predictors_size) ]';
    
    % responses = [ repmat(0, 1, predictors_size) repmat(1, 1, predictors_size) ]';
    % responses = [ repmat(0, 1, predictors_size) ]';
    
    responses = double(responses);
    
    train = data(:, 1:predictors_size);
    % train = data;
    test = data(:, predictors_size + 1:end);

    train_r = responses(1:predictors_size);
    % train_r = responses;
    test_r = responses(predictors_size + 1:end);
    
    k = 20;
    
    correct = 0;
    
    for i = 1:size(test, 2)
        
       disp(i);
                  
       nearest_values = ones(k, 1);
       nearest_labels = zeros(k, 1);
               
       for j = 1:size(train, 2)

           %disp(i)
           %disp(j);
           
           v = norm(train(:, j) - test(:, i));           
           r = train_r(j);
           
           if j == 1
               
               nearest_values = ones(k, 1) * v;
               nearest_labels = ones(k, 1) * r;
               
           else
               
               [mv, mi] = max(nearest_values);
               if v < mv
                   nearest_values(mi) = v;
                   nearest_labels(mi) = r;
               end
               
           end
           
       end
       
       r = mean(nearest_labels);
       
       if r >= 0.5
           r = 1;
       else
           r = 0;
       end
           
       tr = test_r(i);
       
       if tr == r
           correct = correct + 1.0;
       end
       
        
    end
    
    correct = correct / test_size;
    
    disp(correct);

    