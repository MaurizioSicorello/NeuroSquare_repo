% simply uses the nested cv_svr algorithm implemented in canlab, repeats it
% for reliability and collects the correct results for the main loop


function [r, fullOutputs] = cv_svr_repeat(data, numFolds, repeats, subject_id)
    
    % create vector and structure to collect output
    repeated_cv_results = zeros(repeats, 1);
    collectOutputs = struct;
    
    % repetition loop
    for i = 1:repeats
        
        % create fold labels
        kfolds = numFolds;
        cv_outer = cvpartition2(ones(size(data.dat,2),1), 'KFOLD', kfolds, 'Stratify', subject_id);

        cv_results = zeros(numFolds,2);
        
        % k-fold cross-validation
        for j = 1:numFolds
            
            % split training and test data
            trainDat = get_wh_image(data, find(cv_outer.training(j)));
            testDat = get_wh_image(data, find(cv_outer.test(j)));
            
            % find optimal hyperparameter within inner loop          
            hyperpar_num = cv_svr_optPar(trainDat);
            
            % train with optimal hyerpar on whole training set
            [~, stats2, ~] = predict(trainDat, 'cv_svr', 'nfolds', 1, 'C', hyperpar_num);
            pattern_exp_values = transpose(testDat.dat)*stats2.weight_obj.dat;
            cv_results(j,1) = corr(testDat.Y, pattern_exp_values);
            cv_results(j,2) = hyperpar_num;
        
        end
        
        r = tanh(mean(atanh(cv_results(:,1))));
    
        r_ind = cv_results(:,1);
        hyp = cv_results(:,2);
        foldwise = [r_ind, hyp];

        fullOutput = struct('Dataset', inputname(1), 'Algorithm', 'cv_svr', 'numFolds', numFolds, 'corr', r, 'foldwise', foldwise);
        
        repeated_cv_results(i) = r;
        collectOutputs.(sprintf('Repeat%d', i)) = fullOutput;


    end
        
    r = tanh(mean(atanh(repeated_cv_results(:,1))))
    fullOutputs = collectOutputs;
    
end




