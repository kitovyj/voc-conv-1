function analyze()

    load('statistics.mat');

    avg_durations = [0, 0];
    avg_freq = [0, 0];
    
    for i = 1:numel(durations)
        class_durations = durations{i};
        fprintf('total: %d\n', numel(class_durations));            
        avg_durations(i) = mean(class_durations);
        fprintf('average duration: %f\n', avg_durations(i));    
        [h, p] = ttest(class_durations);
        fprintf('student test: %f, %f\n', h, p);    
        class_frequencies = frequencies{i};
        avg_freq(i) = mean(class_frequencies);
        fprintf('average frequency: %f\n', avg_freq(i));        
        [h, p] = ttest(class_frequencies);
        fprintf('student test: %f, %f\n', h, p);    
    end
    
    [h, p] = ttest2(durations{1}, durations{2});
    fprintf('student two sample test for durations : %f, %f\n', h, p);
    [h, p] = ttest2(frequencies{1}, frequencies{2});
    fprintf('student two sample test for frequincies : %f, %f\n', h, p);
        
    % linear models
    
    % merge predictors and responses
    
    % durations{1}(durations{1} > 0.1) = 0.1;
    % durations{2}(durations{2} > 0.1) = 0.1;
        
    predictors = [ durations{1} durations{2} ];
    predictors_size = numel(durations{1});
    predictors_matrix = [ predictors; ones(1, length(predictors)) ]';
    
    responses = [ repmat(0, 1, predictors_size) repmat(1, 1, predictors_size) ]';
    
    b = regress(responses, predictors_matrix);
    
    xx = linspace(0, 0.4, 100);
    yy = b(2)+ b(1).*xx;

    plot(xx, yy, 'r', durations{1}, repmat(0, 1, predictors_size), 'go', durations{2}, repmat(1, 1, predictors_size), 'bo');
    
    grid on

    yy = b(2)+ b(1).*predictors;
    yy(yy < 0.5) = 0;
    yy(yy >= 0.5) = 1;
    correct = (yy == (responses'));
    accuracy = mean(correct);
    
    fprintf('accuracy : %f\n', accuracy);

    % --------------------------------------------------------
    
    predictors = [ frequencies{1} frequencies{2} ];
    predictors_size = numel(durations{1});
    predictors_matrix = [ predictors; ones(1, length(predictors)) ]';
    
    responses = [ repmat(0, 1, predictors_size) repmat(1, 1, predictors_size) ]';
    
    b = regress(responses, predictors_matrix);
    
    %{
    xx = linspace(0, 0.4, 100);
    yy = b(2)+ b(1).*xx;

    plot(xx, yy, 'r', durations{1}, repmat(0, 1, predictors_size), 'go', durations{2}, repmat(1, 1, predictors_size), 'bo');
    
    grid on
    %}

    yy = b(2)+ b(1).*predictors;
    yy(yy < 0.5) = 0;
    yy(yy >= 0.5) = 1;
    correct = (yy == (responses'));
    accuracy = mean(correct);
    
    fprintf('accuracy : %f\n', accuracy);

    % --------------------------------------------------------
    
    predictors = [ loudnesses{1} loudnesses{2} ];
    predictors_size = numel(loudnesses{1});
    predictors_matrix = [ predictors; ones(1, length(predictors)) ]';
    
    responses = [ repmat(0, 1, predictors_size) repmat(1, 1, predictors_size) ]';
    
    b = regress(responses, predictors_matrix);
    
    %{
    xx = linspace(0, 0.4, 100);
    yy = b(2)+ b(1).*xx;

    plot(xx, yy, 'r', durations{1}, repmat(0, 1, predictors_size), 'go', durations{2}, repmat(1, 1, predictors_size), 'bo');
    
    grid on
    %}

    yy = b(2)+ b(1).*predictors;
    yy(yy < 0.5) = 0;
    yy(yy >= 0.5) = 1;
    correct = (yy == (responses'));
    accuracy = mean(correct);
    
    fprintf('accuracy : %f\n', accuracy);
    
    % --------------------------------------------------------
    
    predictor1 = [ durations{1} durations{2} ];
    predictor2 = [ frequencies{1} frequencies{2} ];
    predictor3 = [ loudnesses{1} loudnesses{2} ];
    predictors_size = numel(durations{1});
    predictors_matrix = [ predictor1; predictor2; predictor3; ones(1, length(predictors)) ]';
    
    responses = [ repmat(0, 1, predictors_size) repmat(1, 1, predictors_size) ]';
    
    b = regress(responses, predictors_matrix);
    
    %{
    xx = linspace(0, 0.4, 100);
    yy = b(2)+ b(1).*xx;

    plot(xx, yy, 'r', durations{1}, repmat(0, 1, predictors_size), 'go', durations{2}, repmat(1, 1, predictors_size), 'bo');
    
    grid on
    %}

    yy = b(4)+ (b(3).*predictor3) + (b(2).*predictor2) + (b(1).*predictor1);
    yy(yy < 0.5) = 0;
    yy(yy >= 0.5) = 1;
    correct = (yy == (responses'));
    accuracy = mean(correct);
    
    fprintf('accuracy : %f\n', accuracy);
   
    yy = (predictors_matrix * b)';
    yy(yy < 0.5) = 0;
    yy(yy >= 0.5) = 1;
    correct = (yy == (responses'));
    accuracy = mean(correct);
    
    fprintf('accuracy : %f\n', accuracy);    
    
    figure
    bins = linspace(0, 0.2, 200);
    [h] = hist(durations{1}, bins);
    plot(bins, h, '-', 'Color', 'red');
    hold on;
    [h] = hist(durations{2}, bins);
    plot(bins, h, '-', 'Color', 'blue');

    figure;
    bins = linspace(0, 233, 233);
    [h] = hist(frequencies{1}, bins);
    plot(bins, h, '-', 'Color', 'red');
    hold on;
    [h] = hist(frequencies{2}, bins);
    plot(bins, h, '-', 'Color', 'blue');
    
    % make data to be the same in its length
    %{
    min_el = min(numel(durations{1}), numel(durations{2}));
    
    for i = 1:numel(durations)
    vector = vector(31:end);
    %}
