cd('Results/Multiverse_Loop')

% list model names in results folder
modelNames = dir('ER_LookDiff*.mat');

% check dimensions of model results
testModel = load(modelNames(1).name).fullOutputs;
nRepeats = length(fieldnames(testModel));
nFolds = getfield(testModel, 'Repeat1').numFolds

% create empty df
resultsdf = zeros(length(modelNames)*nRepeats*nFolds, 1);
namesdf = strings(length(modelNames)*nRepeats*nFolds, 1);
count = 0;

% loop to save results in resultsdf
for i = 1:length(modelNames)
    
    % loop through models
    resultsModel = struct2cell(load(modelNames(i).name).fullOutputs);
    
    % loop through repeats
    for j = 1:nRepeats
        
        % loop trough folds
        for k = 1:nFolds
            count = count + 1;
            
            resultsdf(count) = resultsModel{j,1}.foldwise(k,1);
            namesdf(count) = string(modelNames(i).name);
            
            count
        end
        
    end
    
end


resultsTable = table(resultsdf, namesdf);

boxplot(resultsdf, namesdf)

