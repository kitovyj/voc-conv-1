function [Model, Residuals, yPred, Indices] = regressCV(y,X,varargin)

P = parsePairs(varargin);
checkField(P,'NCV',10);
checkField(P,'Method','Linear');
checkField(P,'Constant',1);
checkField(P,'Lambda',0.001);

Indices = {};

NSamples = size(X,1);
if P.Constant; X = [X,ones(NSamples,1)]; end
NPredictors = size(X,1);

NTest = NSamples/P.NCV; 
Bounds = [0,round([NTest:NTest:NSamples])];
if length(unique(Bounds)) < length(Bounds)
  error('The number of crossvalidation iterations is quite high compared to the number of samples');
end

% LOOP OVER CROSS VALIDATION SUBSETS
for iCV = 1:length(Bounds)-1
  fprintf([num2str(iCV),' '])
  % DEFINE TRAIN AND TEST SETS
  cIndTest = [Bounds(iCV)+1 : Bounds(iCV+1)];
  cIndTrain = setdiff([1:NSamples],cIndTest);
  
  % ESTIMATE
  switch P.Method
    case 'Linear';
      Model.b(iCV,:) = regress(y(cIndTrain),X(cIndTrain,:));
    case 'Ridge';
      %  Model.b(iCV,:) = ridge(y(cIndTrain),X(cIndTrain,:), P.Lambda);
      RR = X(cIndTrain,:)'*X(cIndTrain,:);  RS = X(cIndTrain,:)'*y(cIndTrain);
      Model.b(iCV,:) = inv(RR + P.Lambda*eye(size(RR)))*RS;
    case 'SVM';
      options.MaxIter = Inf;        
      Model.SVM(iCV)= svmtrain(X(cIndTrain,:),y(cIndTrain), 'Options', options);
      %Model.SVM(iCV)= svmtrain(X(cIndTrain,:), y(cIndTrain), 'kernel_function', 'quadratic', 'boxconstraint', 1, 'tolkkt', 0.1);
  end

  % PREDICT 
  switch P.Method
    case 'Ridge';
      yPred(cIndTest) = X(cIndTest,:)*Model.b(iCV,:)';
    case 'Linear';
      yPred(cIndTest) = X(cIndTest,:)*Model.b(iCV,:)';
    case 'SVM';
      yPred(cIndTest) = svmclassify(Model.SVM(iCV),X(cIndTest,:));
  end

  % ASSESS ERROR
  Residuals(cIndTest) = y(cIndTest) - yPred(cIndTest)';
  
  Indices{end + 1} = cIndTest;
  
end

fprintf('\n');