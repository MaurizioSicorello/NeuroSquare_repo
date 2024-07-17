
function [optPar] = cv_svr_optPar(data)
    
    gridResolution = 10;
    nFold = 5;
    paramGrid = linspace(eps,5,gridResolution);
    
    accResults = zeros(1, 10);
    
    for i = 1:length(paramGrid)
        
        [cverr, stats] = predict(data, 'cv_svr', 'C', paramGrid(i));
        accResults(i) = stats.pred_outcome_r;
        
    end
    
    [~, index] = max(accResults);
    optPar = paramGrid(index)

    
end
