% NESTED CROSS-VALIDATION WITH BAYESIAN PARAMETER OPTIMIZATION

% INPUTS:
% data: fmri_data object
% ML_alg: name of machine learning algorithm to use in the canlab predict
% function
% min: Minimum of hyperparameter space to search
% max: Minimum of hyperparameter space to search
% num_res: numeric resolution of hyperparameter space (e.g. integer for pls)
% numFolds: number of outer folds

% OUTPUTS:
% r: mean cross-validated correlation between predicted and observed
% foldwise: individual cross-validated correlations and hyperparameters for outer fold


% NOTE: fMRI_data object must currently contain subject IDs in yourfMRI_data.metadata_table.subject_id

% POSSIBLE FUTURE CHANGES: 
% -probably start worker pool outside loop
% warning message for empty subject_id field. maybe fill field
% automatically then


% Function for repeated nested cross-validation
function [r, fullOutputs] = nestedCrossValRepeat(data, ML_alg, min, max, num_res, numFolds, repeats, subject_id, numWorkers)
    
    if numWorkers > 0
        % start parallel pool
        if ~isempty(gcp('nocreate'))
                delete(gcp('nocreate'));
        end
        parpool(numWorkers)
    end

    % create vector and structure to collect outputs
    repeated_cv_results = zeros(repeats, 1);
    collectOutputs = struct;

    % repetion loop
    for h = 1:repeats
        
        [r, foldwise, fullOutput] = nestedCrossVal(data, ML_alg, min, max, num_res, numFolds, subject_id);
        repeated_cv_results(h) = r;
        collectOutputs.(sprintf('Repeat%d', h)) = fullOutput;

    end
    
    r = tanh(mean(atanh(repeated_cv_results(:,1))))
    fullOutputs = collectOutputs;
    
    delete(gcp('nocreate'))

end



% Function for nested cross-validation
function [r, foldwise, fullOutput] = nestedCrossVal(data, ML_alg, min, max, num_res, numFolds, subject_id)

    % create fold labels
    kfolds = numFolds;
    cv_outer = cvpartition2(ones(size(data.dat,2),1), 'KFOLD', kfolds, 'Stratify', subject_id);

    cv_results = zeros(numFolds,2);

    for i = 1:kfolds

        % split training and test data
        trainDat = get_wh_image(data, find(cv_outer.training(i)));
        testDat = get_wh_image(data, find(cv_outer.test(i)));
        
        subject_id_inner = subject_id(cv_outer.training(i));

        % estimate optimal hyperparameter in an inner cv-loop
        optNumComp = optHyperpar(trainDat, ML_alg, min, max, num_res, subject_id_inner).XAtMinObjective{1,1};

        % use optimal hyperparameter to build model on whole training data and
        % apply to test data
        [cverr, stats, optout] = predict(trainDat, 'algorithm_name', ML_alg, 'numcomponents', optNumComp, 'nfolds', 1);
        pattern_exp_values = transpose(testDat.dat)*stats.weight_obj.dat;
        cv_results(i,1) = corr(testDat.Y, pattern_exp_values);
        cv_results(i,2) = optNumComp;

    end
    
    r = tanh(mean(atanh(cv_results(:,1))));
    
    r_ind = cv_results(:,1);
    hyp = cv_results(:,2);
    foldwise = [r_ind, hyp];
    
    fullOutput = struct('Dataset', inputname(1), 'Algorithm', ML_alg, 'numFolds', numFolds, 'corr', r, 'foldwise', foldwise);

    
end



% main function for parameter optimization
function optimalNcomp = optHyperpar(dat, ML_alg, min, max, num_res, subject_id_inner)
   dims_bt = optimizableVariable('btDims', [min, max], 'Type', num_res);
   constraint=@(x1)(x1.btDims > 0);
   objfxn = @(dims1)(lossEst(dims1, dat, ML_alg, subject_id_inner));

   optimalNcomp = bayesopt(objfxn,[dims_bt],'XConstraintFcn',constraint,...
        'AcquisitionFunctionName','expected-improvement-plus','MaxObjectiveEvaluations',30,...
        'UseParallel',0, 'PlotFcn', []);
end 


% loss function for parameter optimization
function loss = lossEst(dim, dat, ML_alg, subject_id_inner)
    % we want to incorporate CV fold slicing variance into our estimator so
    % let's get new CV folds to use on this iteration. If we revisit this
    % spot in the search space we'll get new slices and the variance can be
    % incorporated by bayesopt into its model of the loss function
    
    kfolds = 5;
    cv_inner = cvpartition2(ones(size(dat.dat,2),1), 'KFOLD', kfolds, 'Stratify', subject_id_inner);

    [I,J] = find([cv_inner.test(1),cv_inner.test(2), cv_inner.test(3), cv_inner.test(4), cv_inner.test(5)]);
    fold_labels_inner = sortrows([I,J]);
    fold_labels_inner = fold_labels_inner(:,2);
    
    r = dat.predict('algorithm_name', ML_alg, 'numcomponents', dim.btDims, 'nfolds', fold_labels_inner, ...
        'verbose',0, 'error_type','r');
    
    loss = 1-r;
    
end

