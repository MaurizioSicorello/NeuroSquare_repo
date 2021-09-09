% contains the subroutine for hyperparameter estimation from
% nestedCrossValRepeat to only estimate the optimal hyperparameter and
% refit the model using this parameter

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

