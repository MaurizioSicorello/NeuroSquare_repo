%%%%%%%%%%%% 
% PREPARE

% load data
image_names = filenames(fullfile(pwd, 'Data\changData\', char("*.nii")), 'absolute');
fmriData_full = fmri_data(image_names);

% extract subject IDs and ratings
splitNames = cellfun(@(x) strsplit(strtrim(x), '_'), cellstr(fmriData_full.image_names), 'UniformOutput', false);
IDs = cellfun(@(x) x{3}, splitNames, 'UniformOutput', false);
ratings = cellfun(@(x) str2double(x{5}), splitNames);
fmriData_full.Y = ratings;

% create CV folds
IDs_unique = unique(IDs)
foldInd = repelem([1:10], 15);
foldInd = foldInd(1:length(IDs_unique))

% preprocess images
gray_mask = fmri_mask_image('gray_matter_mask.img');
fmriData_full = fmriData_full.apply_mask(gray_mask);
fmriData_full.Y = zscore(fmriData_full.Y);
fmriData_full = rescale(fmriData_full, 'zscorevoxels')
fmriData_full = rescale(fmriData_full, 'centerimages');


%%%%%%%%%%%% 
% PREDICT

% results vector
corrOut = zeros(length(IDs_unique), 1);
count = 0

% loop through cv procedure (10-fold)
for i=1:10
    
    % subset training fold
    testIDs_temp = IDs_unique(foldInd == i)
    foldInd_temp = ismember(IDs, testIDs_temp)
    trainDat_temp = get_wh_image(fmriData_full, ~foldInd_temp)
    testDat_temp = get_wh_image(fmriData_full, foldInd_temp)

    % train model
    [cverr, stats, optional_outputs] = predict(trainDat_temp, 'cv_svr', 'C', 1.6667, 'nfolds', 1)

    % predict test data
    [pattern_exp_values] = apply_mask(testDat_temp, stats.weight_obj, 'pattern_expression', 'ignore_missing');
    %corrOut(i) = corr(pattern_exp_values, testDat_temp.Y)

    % calculate within-person correlations
    for j=1:length(testIDs_temp)
        
        count = count+1;
        testPersonInd_temp = ismember(IDs(foldInd_temp), testIDs_temp(j));

        if sum(testPersonInd_temp) <= 3

            corrOut(count) = NaN;

        elseif length(unique(testDat_temp.Y(testPersonInd_temp))) == 1

            corrOut(count) = NaN;

        else

            corrOut(count) = corr(pattern_exp_values(testPersonInd_temp), testDat_temp.Y(testPersonInd_temp));

        end

    end

end


nanmean(corrOut)
nanstd(corrOut)

