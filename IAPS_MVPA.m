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

% test prediction [maybe important: predict appears to center the outcome]
[cverr, stats, optional_outputs] = predict(IAPS_train, 'algorithm_name', 'cv_pcr', 'numcomponents', 200, 'nfolds', 1);
% [pattern_exp_values] = apply_mask(IAPS_train, stats.weight_obj, 'pattern_expression', 'ignore_missing')
% corr(IAPS_train.Y, pattern_exp_values)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% predict on raw data

% use main function on raw data [error in apply_mask for some reason. works
% after the rescale function is applied]
[r, hyp] = nestedCrossVal(IAPS_train, 'cv_pls', 0, 2, 'integer', 4, 5)



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% predict on cross-sectionally z-scored data

% standardize IVs and DVs cross-sectionally
% performance even worse after scaling. don't understand why
IAPS_train_z = rescale(IAPS_train, 'zscorevoxels');
IAPS_train.Y = zscore(IAPS_train_z.Y);
%predict
[r, hyp] = nestedCrossVal(IAPS_train_z, 'cv_pls', 0, 2, 'integer', 4, 5)



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% predict on cross-sectionally and image-wise z-scored data

% image-wise standardization
IAPS_train_z2 = rescale(IAPS_train_z, 'zscoreimages');
% predict
[r, hyp] = nestedCrossVal(IAPS_train_z, 'cv_pls', 0, 2, 'integer', 4, 5)


