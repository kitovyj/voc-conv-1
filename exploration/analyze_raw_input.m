function [female, male, overall] = analyze_raw_input(data,responses,varargin)

P = parsePairs(varargin);
checkField(P,'Method','Ridge');
checkField(P,'Balance',0);
checkField(P,'Reduce',4);


    female = [];
    male = [];
    overall = [];
    
    if P.Reduce; data = data(P.Reduce:P.Reduce:end,:);     end
    
    if P.Balance
      N1 = sum(responses==1);
      Ind0 = find(responses==0);
      Ind1 = find(responses==1);
      Ind0 = Ind0(1:N1);
      IndSel = [Ind0;Ind1];
      IndSel = IndSel(randperm(length(IndSel)));
      data = data(:, IndSel);
      responses = responses(IndSel);
    end
    
    % SVMModel = fitcsvm(X,Y,'KernelFunction','rbf','Standardize',true,'ClassNames', {'negClass', 'posClass'});
    
    %test_size = 500;
    test_size = 0;
    predictors_size = size(data, 2) - test_size;
    
    responses = double(responses);
       
    [Model, Residuals, yPred, indices] = regressCV(responses, data(:, 1:predictors_size)', 'Method', P.Method);
    
    yy = yPred;
    yy(yy < 0.5) = 0;
    yy(yy >= 0.5) = 1;
       
    %correct = (yy == (responses(predictors_size + 1:end)'));
        
    female_indices = find(responses == 0);
    male_indices = find(responses == 1);
       
    correct = (yy(female_indices) == responses(female_indices)');    
    accuracy_female = mean(correct);

    correct = (yy(male_indices) == responses(male_indices)');    
    accuracy_male = mean(correct);

    accuracy = mean([accuracy_male, accuracy_female]);

    fprintf('accuracy(female): %f\n', accuracy_female);    
    fprintf('accuracy(male) : %f\n', accuracy_male);    
    fprintf('accuracy(overall) : %f\n', accuracy);
    
    for k = 1:length(indices)

        i = indices{k};
        y = yy(i);
        r = responses(i);
        
        female_indices = find(r == 0);
        male_indices = find(r == 1);

        correct = (y(female_indices) == r(female_indices)');    
        accuracy_female = mean(correct);

        correct = (y(male_indices) == r(male_indices)');    
        accuracy_male = mean(correct);

        accuracy = mean([accuracy_male, accuracy_female]);

        female = [female accuracy_female];
        male = [male accuracy_male];
        overall = [overall accuracy];
        
        fprintf('accuracy(female %d): %f\n', k, accuracy_female);    
        fprintf('accuracy(male %d) : %f\n', k, accuracy_male);    
        fprintf('accuracy(overall %d) : %f\n', k, accuracy);
        
    end
    