function analyze_raw_input()

    load('statistics.mat');

    % linear model
    
    data = raw;
    
    test_size = 500;
    predictors_size = size(data, 2) - test_size;
    predictors_matrix = [ data(:, 1:predictors_size); ones(1, predictors_size) ]';
    
    % responses = [ repmat(0, 1, predictors_size) repmat(1, 1, predictors_size) ]';
    % responses = [ repmat(0, 1, predictors_size) ]';
    
    responses = double(responses);
    
    b = regress(responses(1:predictors_size), predictors_matrix);
    
    %{
    xx = linspace(0, 0.4, 100);
    yy = b(2)+ b(1).*xx;

    plot(xx, yy, 'r', durations{1}, repmat(0, 1, predictors_size), 'go', durations{2}, repmat(1, 1, predictors_size), 'bo');
    
    grid on
    %}
    
    test_matrix = [ data(:, predictors_size + 1:end); ones(1, test_size) ]';
    
    yy = (test_matrix * b)';
    yy(yy < 0.5) = 0;
    yy(yy >= 0.5) = 1;
    correct = (yy == (responses(predictors_size + 1:end)'));
    accuracy = mean(correct);
    
    fprintf('accuracy : %f\n', accuracy);

    