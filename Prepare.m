%%%%%%%% Analyses on IAPS paradigm %%%%%%%% 

% dependencies: spm, canlabcoretools, canlab_single_trials
% put function folder on path
% go to directory 'subject-level-maps' as the base directory

% example image for the paradigm of interest
betaName = 'Subject_004_IAPS_LookNeg-vs-LookNeut.nii.gz';

% locate image
myfile = which(betaName);
mydir = fileparts(myfile);
if isempty(mydir), disp('Uh-oh! I can''t find the data.'), else disp('Data found.'), end

% load IAPS files into fmri_data object
image_names = filenames(fullfile(mydir, '*IAPS_LookNeg-vs-LookNeut.nii.gz'), 'absolute');
IAPS_all = fmri_data(image_names);

% check descriptives/outliers
%descriptives(IAPS_all);
%plot(IAPS_all)
%[ds, expectedds, p, wh_outlier_uncorr, wh_outlier_corr] = mahal(IAPS_all, 'noplot', 'corr'); % no bonferroni-corrected outliers. identical with bonferroni-holm

% make list of unpadded fMRI IDs
[P, N, E] = cellfun(@fileparts, image_names, 'UniformOutput', false);
id_fMRI = extractBetween(N, 9, 11);
id_fMRI = str2double(id_fMRI);
id_fMRI = array2table(id_fMRI, 'VariableNames', {'id'});
IAPS_all.metadata_table.subject_id = id_fMRI;

% load questionnaire data
AHAB2_quest = readtable('../AHAB2_psychVars_deidentified');
PIP_quest = readtable('../PIP_psychVars_deidentified');

% merge data
PIP_quest.Properties.VariableNames{find(string(PIP_quest.Properties.VariableNames) == "neoN")} = 'NEON';
NEON_all = [AHAB2_quest(:,{'id', 'NEON'}); PIP_quest(:,{'id', 'NEON'})];
fMRI_NEON = join(id_fMRI, NEON_all);

% plot neuroticism
histogram(fMRI_NEON.NEON)
boxplot(fMRI_NEON.NEON)

%store Neuroticism in fmri_object and subset complete cases
IAPS_all.Y = fMRI_NEON.NEON;
completeCases = ~isnan(fMRI_NEON.NEON);
IAPS_all_compl = get_wh_image(IAPS_all, completeCases);

% mask grey matter
gray_mask = fmri_mask_image('gray_matter_mask.img');
IAPS_all_compl = IAPS_all_compl.apply_mask(gray_mask);

% create stratified hold-out set 
%(not implemented in matlab for specific hold-out N and continuous vars)
cd('../holdout-identifiers')
writematrix(IAPS_all_compl.Y, 'N_IAPS')
holdoutIndex = readtable('N_IAPS_holdoutIndex');
cd('../Subject-level-maps')
IAPS_train = get_wh_image(IAPS_all_compl, xor(holdoutIndex.trainIndex_bin, 0));
IAPS_test = get_wh_image(IAPS_all_compl, xor(holdoutIndex.testIndex_bin, 0));

% standardize IVs and DVs cross-sectionally
% performance even worse after scaling. don't understand why
IAPS_train = rescale(IAPS_train, 'zscorevoxels');
IAPS_train.Y = zscore(IAPS_train.Y);

% predict
[cverr, stats, optional_outputs] = predict(IAPS_train, 'algorithm_name', 'cv_pls', 'numcomponents', 50);

% image-wise standardization
IAPS_train_z = rescale(IAPS_train, 'zscoreimages');

% predict
[cverr, stats, optional_outputs] = predict(IAPS_train_z, 'algorithm_name', 'cv_pls', 'numcomponents', 50);


% TO DO: build nested PLS code with parameter optimization

% get number of subjects and subject id
[~,~,subject_id] = unique(IAPS_train_z.metadata_table.subject_id,'stable');
uniq_subject_id = unique(subject_id);
n_subj = length(uniq_subject_id);

% make indicator for stratified hold-out set [bogdan's code]
kfolds = 5;
cv = cvpartition2(ones(size(IAPS_train_z.dat,2),1), 'KFOLD', kfolds, 'Stratify', subject_id);
[I,J] = find([cv.test(1),cv.test(2), cv.test(3), cv.test(4), cv.test(5)]);
fold_labels = sortrows([I,J]);
fold_labels = fold_labels(:,2);

% define hyperparameter search space by finding maximum df you'll have for
% any training fold
max_subj_per_fold = length(IAPS_train_z.Y) - floor(length(IAPS_train_z.Y)/5);
min_subj_per_fold = length(IAPS_train_z.Y) - ceil(length(IAPS_train_z.Y)/5);
minTrainingSize = length(fold_labels) - max(accumarray(fold_labels,1));

% optimize number of components
x = optHyperpar(IAPS_train_z, 'cv_pls', 0, min_subj_per_fold - 1, 'integer', 4).XAtMinObjective{1,1};
x = optHyperpar(IAPS_train_z, 'cv_pls', 0, 3, 'integer', 4).XAtMinObjective{1,1};


% k-fold cross-validation
kfolds = 5;
cv = cvpartition2(ones(size(IAPS_train_z.dat,2),1), 'KFOLD', kfolds, 'Stratify', subject_id);
[I,J] = find([cv.test(1),cv.test(2), cv.test(3), cv.test(4), cv.test(5)]);
fold_labels = sortrows([I,J]);
fold_labels = fold_labels(:,2);

cv_results = zeros(5,2);

for i = 1:kfolds

    % split training and test data
    trainDat = get_wh_image(IAPS_train_z, fold_labels ~= i);
    testDat = get_wh_image(IAPS_train_z, fold_labels == i);
    
    % estimate optimal hyperparameter in an inner cv-loop
    optNumComp = optHyperpar(trainDat, 'cv_pls', 0, 3, 'integer', 4).XAtMinObjective{1,1};
    
    % use optimal hyperparameter to build model on whole draining data and
    % apply to test data
    [cverr, stats, optout] = predict(trainDat, 'algorithm_name', 'cv_pls', 'numcomponents', optNumComp, 'nfolds', 1);
    pattern_weights = stats.weight_obj;
    [pattern_exp_values] = apply_mask(testDat, pattern_weights, 'pattern_expression', 'ignore_missing');
    cv_results(i,1) = corr(testDat.Y, pattern_exp_values);
    cv_results(i,2) = optNumComp;
    
end

% average fisher-transformed correlation accuracy and transform back to r
tanh(mean(atanh(cv_results(:,1))))

[r, hyp] = nestedCrossVal(IAPS_train_z, 'cv_pls', 0, 3, 'integer', 4, 5)