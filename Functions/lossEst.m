% loss function for parameter optimization
function loss = lossEst(dim, dat)
    subject_id = dat.metadata_table.subject_id;
    [~,~,subject_id] = unique(subject_id);
    % we want to incorporate CV fold slicing variance into our estimator so
    % let's get new CV folds to use on this iteration. If we revisit this
    % spot in the search space we'll get new slices and the variance can be
    % incorporated by bayesopt into its model of the loss function
    kfolds = 5;
    cv = cvpartition2(ones(size(dat.dat,2),1), 'KFOLD', kfolds, 'Stratify', subject_id);

    [I,J] = find([cv.test(1),cv.test(2), cv.test(3), cv.test(4), cv.test(5)]);
    fold_labels = sortrows([I,J]);
    fold_labels = fold_labels(:,2);
    
    r = dat.predict('algorithm_name', 'cv_pls', 'numcomponents', dim.btDims, 'nfolds', fold_labels, ...
        'verbose',0, 'subjIDs', subject_id, 'error_type','r');

%     r = dat.predict('algorithm_name','cv_mlpcr',...
%            'nfolds',fold_labels,'numcomponents',[dim.btDims, dim.wiDims], ...
%            'verbose',0, 'subjIDs', subject_id, 'error_type','r');
    loss = 1-r;
end

