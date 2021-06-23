% CROSS-Validated random forest regression

% INPUTS
% data: fmri_data object
% numFolds: number of cross-validation folds
% repeats: How often to repeat procedure
% PCA: If true, do PCA analysis before random forest, extracting k = n
% components

% OUTPUTS
% r: mean cross-validated correlation between predicted and observed
% foldwise: individual cross-validated correlations and hyperparameters for outer fold


function [r, fullOutputs] = cv_ranfor_repeat(data, numFolds, repeats, numTrees, numWorkers, PCA_prepro, subject_id)

    % create vector and structure to collect outputs
    repeated_cv_results = zeros(repeats, 1);
    collectOutputs = struct;
    
    % repetion loop
    for i = 1:repeats
        
        [r, foldwise, fullOutput] = cv_ranfor(data, numFolds, numTrees, numWorkers, PCA_prepro, subject_id);
        repeated_cv_results(i) = r;
        collectOutputs.(sprintf('Repeat%d', i)) = fullOutput;
        
    end
    
    r = tanh(mean(atanh(repeated_cv_results(:,1))));
    fullOutputs = collectOutputs;
    
end


function [r, foldwise, fullOutput] = cv_ranfor(data, numFolds, numTrees, numWorkers, PCA_prepro, subject_id)

    % ADD OPTION FOR PCA HERE
    if PCA_prepro == true
        
        [~, score] = pca(data.dat');
        data.dat = score'; 
    end

    % create fold labels
    kfolds = numFolds;
    cv_outer = cvpartition2(ones(size(data.dat,2),1), 'KFOLD', kfolds, 'Stratify', subject_id);

    cv_results = zeros(numFolds,2);
    
    % start parallel pool
    parpool(numWorkers)
    paroptions = statset('UseParallel',true);
    
    % cross-validated random forest
    for j = 1:kfolds
        
        % split training and test data
        trainDat = get_wh_image(data, find(cv_outer.training(j)));
        testDat = get_wh_image(data, find(cv_outer.test(j)));
        
        % train rf on trainset and predict outcome in testset
        rf_model = TreeBagger(numTrees, trainDat.dat', trainDat.Y, 'Method','regression', 'Options', paroptions);
        pattern_exp_values = predict(rf_model, testDat.dat');
        cv_results(j, 1) = corr(testDat.Y, pattern_exp_values);
        cv_results(j, 2) = numTrees; 
        
    end
    
    delete(gcp('nocreate'))
    
    r = tanh(mean(atanh(cv_results(:,1))));
    
    r_ind = cv_results(:,1);
    hyp = cv_results(:,2);
    foldwise = [r_ind, hyp];
    
    fullOutput = struct('Dataset', inputname(1), 'Algorithm', 'cv_ranfor', 'numFolds', numFolds, 'corr', r, 'foldwise', foldwise);
    
end


       