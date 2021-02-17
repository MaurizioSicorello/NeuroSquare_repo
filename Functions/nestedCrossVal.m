% NESTED CROSS-VALIDATION WITH BAYESIAN PARAMETER OPTIMIZATION

% INPUTS:
% data: fmri_data object
% ML_alg: name of machine learning algorithm to use in the canlab predict
% function
% min: Minimum of hyperparameter space to search
% max: Minimum of hyperparameter space to search
% num_res: numeric resolution of hyperparameter space (e.g. integer for pls)
% numWorkers: number of workers for parallel computing
% numFolds: number of outer folds

% OUTPUTS:
% r: cross-validated correlation between predicted and observed
% hyp: optimal hyperparameter determined in inner cross-validation
% outer cross-validation loop

% TO DO: probably start worker pool outside loop


function [r, hyp] = nestedCrossVal(data, ML_alg, min, max, num_res, numWorkers, numFolds)

    % get subject ID from fMRI_data 
    [~,~,subject_id] = unique(data.metadata_table.subject_id,'stable');

    % 
    kfolds = numFolds;
    cv_outer = cvpartition2(ones(size(data.dat,2),1), 'KFOLD', kfolds, 'Stratify', subject_id);
    [I,J] = find([cv_outer.test(1),cv_outer.test(2), cv_outer.test(3), cv_outer.test(4), cv_outer.test(5)]);
    fold_labels_outer = sortrows([I,J]);
    fold_labels_outer = fold_labels_outer(:,2);

    cv_results = zeros(5,2);

    for i = 1:kfolds

        % split training and test data
        trainDat = get_wh_image(data, fold_labels_outer ~= i);
        testDat = get_wh_image(data, fold_labels_outer == i);

        % estimate optimal hyperparameter in an inner cv-loop
        optNumComp = optHyperpar(trainDat, ML_alg, min, max, num_res, numWorkers).XAtMinObjective{1,1};

        % use optimal hyperparameter to build model on whole training data and
        % apply to test data
        [cverr, stats, optout] = predict(trainDat, 'algorithm_name', ML_alg, 'numcomponents', optNumComp, 'nfolds', 1);
        pattern_weights = stats.weight_obj;
        [pattern_exp_values] = apply_mask(testDat, pattern_weights, 'pattern_expression', 'ignore_missing');
        cv_results(i,1) = corr(testDat.Y, pattern_exp_values);
        cv_results(i,2) = optNumComp;

    end
    
    r = tanh(mean(atanh(cv_results(:,1))));
    hyp = cv_results(:,2);
    
end



% main function for parameter optimization
function optimalNcomp = optHyperpar(dat, ML_alg, min, max, num_res, numWorkers)
   dims_bt = optimizableVariable('btDims', [min, max], 'Type', num_res);
   constraint=@(x1)(x1.btDims > 0);
   objfxn = @(dims1)(lossEst(dims1, dat, ML_alg));

   if ~isempty(gcp('nocreate'))
        delete(gcp('nocreate'));
   end
   parpool(numWorkers)
   optimalNcomp = bayesopt(objfxn,[dims_bt],'XConstraintFcn',constraint,...
        'AcquisitionFunctionName','expected-improvement-plus','MaxObjectiveEvaluations',30,...
        'UseParallel',1);
end 


% loss function for parameter optimization
function loss = lossEst(dim, dat, ML_alg)
    subject_id = dat.metadata_table.subject_id;
    [~,~,subject_id] = unique(subject_id);
    % we want to incorporate CV fold slicing variance into our estimator so
    % let's get new CV folds to use on this iteration. If we revisit this
    % spot in the search space we'll get new slices and the variance can be
    % incorporated by bayesopt into its model of the loss function
    kfolds = 5;
    cv_inner = cvpartition2(ones(size(dat.dat,2),1), 'KFOLD', kfolds, 'Stratify', subject_id);

    [I,J] = find([cv_inner.test(1),cv_inner.test(2), cv_inner.test(3), cv_inner.test(4), cv_inner.test(5)]);
    fold_labels_inner = sortrows([I,J]);
    fold_labels_inner = fold_labels_inner(:,2);
    
    r = dat.predict('algorithm_name', ML_alg, 'numcomponents', dim.btDims, 'nfolds', fold_labels_inner, ...
        'verbose',0, 'error_type','r');
    
    loss = 1-r;
    
end

