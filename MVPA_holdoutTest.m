%%%% check random forest procedure on some simulated data!


%%%%%%%%%%%%%%%%%%%%%%%%%%
% load questionnaire data
cd('Data')
AHAB2_quest = readtable('AHAB2_psychVars_deidentified');
PIP_quest = readtable('PIP_psychVars_deidentified');

AHAB2_vars = AHAB2_quest(:,{'id', 'ER_LookDiff', 'pnsx_pa', 'pnsx_na', 'STAI', 'BDI_TOT', 'NEON', 'NEON1', 'NEON2', 'NEON3', 'NEON4', 'NEON5', 'NEON6', 'NEONX'});
PIP_vars = PIP_quest(:,{'id', 'ER_LookDiff', 'PA_rescale', 'NA_rescale', 'Trait_Anxiety', 'BDI_total', 'neoN', 'neoN1', 'neoN2', 'neoN3', 'neoN4', 'neoN5', 'neoN6', 'NEONX_empty'});

All_Y = array2table([AHAB2_vars{:,:}; PIP_vars{:,:}], 'VariableNames', ...
   {'id', 'ER_LookDiff', 'PA', 'NA', 'STAI', 'BDI', 'neoN', 'neoN1', 'neoN2', 'neoN3', 'neoN4', 'neoN5', 'neoN6', 'NEONX'});


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% neuroticism random forest model

%%%%%%%%%%%%%%%%%%%%%%%%%%
% load fMRI dataset

cd('Subject-level-maps')
image_names = filenames(fullfile(pwd, char("*IAPS_LookNeg-vs-Baseline.nii")), 'absolute');
IAPS_LookBase_all = fmri_data(image_names);

% load outcome
neuroticism = All_Y(:,{'id', 'neoN'});

% make list of unpadded fMRI IDs
[P, N, E] = cellfun(@fileparts, image_names, 'UniformOutput', false);
id_fMRI = extractBetween(N, 9, 11);
id_fMRI = str2double(id_fMRI);
id_fMRI = array2table(id_fMRI, 'VariableNames', {'id'});
IAPS_LookBase_all.metadata_table.subject_id = id_fMRI;

% join fMRI IDs with outcome data
fMRI_neuroticism = innerjoin(id_fMRI, neuroticism);
        
% store DV in fmri_object and subset complete cases
IAPS_LookBase_all.Y = fMRI_neuroticism{:,'neoN'};
completeCases = ~isnan(fMRI_neuroticism{:,2});
IAPS_LookBase_all_compl = get_wh_image(IAPS_LookBase_all, completeCases);

% apply grey matter mask
gray_mask = fmri_mask_image('gray_matter_mask.img');
IAPS_LookBase_all_compl = IAPS_LookBase_all_compl.apply_mask(gray_mask);

% z-score outcome
IAPS_LookBase_all_compl.Y = zscore(IAPS_LookBase_all_compl.Y);
        
% z-score voxels
IAPS_LookBase_all_compl = rescale(IAPS_LookBase_all_compl, 'zscorevoxels');

% PCA
[coeffN, scoreN] = pca(IAPS_LookBase_all_compl.dat');
IAPS_LookBase_all_compl.dat = scoreN';

% train test split
cd('../holdout-identifiers')
holdoutIndex = readtable('N_IAPS_holdoutIndex');
cd('../Subject-level-maps')
IAPS_LookBase_train = get_wh_image(IAPS_LookBase_all_compl, xor(holdoutIndex.trainIndex_bin, 0));
IAPS_LookBase_test = get_wh_image(IAPS_LookBase_all_compl, xor(holdoutIndex.testIndex_bin, 0))

% train and test on holdout set
rf_model = TreeBagger(1000, IAPS_LookBase_train.dat', IAPS_LookBase_train.Y, 'Method','regression')
pattern_exp_values = predict(rf_model, IAPS_LookBase_test.dat');
[r, p] = corr(IAPS_LookBase_test.Y, pattern_exp_values)



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% look negative neutral model 

%%%%%%%%%%%%%%%%%%%%%%%%%%
% load fMRI dataset

image_names = filenames(fullfile(pwd, char("*IAPS_LookNeg-vs-LookNeut.nii")), 'absolute');
IAPS_LookNeut_all = fmri_data(image_names);

% load outcome
Look_NegNeut = All_Y(:,{'id', 'ER_LookDiff'});

% make list of unpadded fMRI IDs
[P, N, E] = cellfun(@fileparts, image_names, 'UniformOutput', false);
id_fMRI = extractBetween(N, 9, 11);
id_fMRI = str2double(id_fMRI);
id_fMRI = array2table(id_fMRI, 'VariableNames', {'id'});
IAPS_LookNeut_all.metadata_table.subject_id = id_fMRI;

% join fMRI IDs with outcome data
fMRI_NegNeut = innerjoin(id_fMRI, Look_NegNeut);
        
% store DV in fmri_object and subset complete cases
IAPS_LookNeut_all.Y = fMRI_NegNeut{:,'ER_LookDiff'};
completeCases = ~isnan(fMRI_NegNeut{:,2});
IAPS_LookNeut_all_compl = get_wh_image(IAPS_LookNeut_all, completeCases);

% apply grey matter mask
gray_mask = fmri_mask_image('gray_matter_mask.img');
IAPS_LookNeut_all_compl = IAPS_LookNeut_all_compl.apply_mask(gray_mask);

% z-score outcome
IAPS_LookNeut_all_compl.Y = zscore(IAPS_LookNeut_all_compl.Y);
        
% z-score voxels
IAPS_LookNeut_all_compl = rescale(IAPS_LookNeut_all_compl, 'zscorevoxels');

% z-score images
IAPS_LookNeut_all_compl = rescale(IAPS_LookNeut_all_compl, 'zscoreimages');

% train test split
cd('../holdout-identifiers')
holdoutIndex = readtable('N_IAPS_holdoutIndex');
cd('../Subject-level-maps')
IAPS_LookNeut_train = get_wh_image(IAPS_LookNeut_all_compl, xor(holdoutIndex.trainIndex_bin, 0));
IAPS_LookNeut_test = get_wh_image(IAPS_LookNeut_all_compl, xor(holdoutIndex.testIndex_bin, 0));

% train and test on holdout set
% [~,~,subject_id] = unique(IAPS_LookNeut_train.metadata_table.subject_id,'stable')
% maxComp = floor(length(subject_id)*4/5)-1
% optComp = optHyperpar(IAPS_LookNeut_train, 'cv_pls', 1, maxComp, 'integer', subject_id) % optimal comps: 22
% [cverr, stats, optional_outputs] = predict(IAPS_LookNeut_train, 'cv_pls', 'numcomponents', table2array(optComp.XAtMinEstimatedObjective))
% cd('../../Results/TrainingPatterns')
% write(stats.weight_obj, 'fname', 'NegNeutRatings.nii');

cd('../../Results/TrainingPatterns')
NegNeutMask = fmri_data('NegNeutRatings.nii');
% apply pattern to data
[pattern_exp_values] = apply_mask(IAPS_LookNeut_test, NegNeutMask, 'pattern_expression', 'ignore_missing');
[r, p] = corr(IAPS_LookNeut_test.Y, pattern_exp_values)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STAI random forest model

%%%%%%%%%%%%%%%%%%%%%%%%%%
% load fMRI dataset

cd('Subject-level-maps')
image_names = filenames(fullfile(pwd, char("*IAPS_LookNeg-vs-Baseline.nii")), 'absolute');
IAPS_LookBase_all = fmri_data(image_names);

% load outcome
STAI = All_Y(:,{'id', 'STAI'});

% make list of unpadded fMRI IDs
[P, N, E] = cellfun(@fileparts, image_names, 'UniformOutput', false);
id_fMRI = extractBetween(N, 9, 11);
id_fMRI = str2double(id_fMRI);
id_fMRI = array2table(id_fMRI, 'VariableNames', {'id'});
IAPS_LookBase_all.metadata_table.subject_id = id_fMRI;

% join fMRI IDs with outcome data
fMRI_STAI = innerjoin(id_fMRI, STAI);
        
% store DV in fmri_object and subset complete cases
IAPS_LookBase_all.Y = fMRI_STAI{:,'STAI'};
completeCases = ~isnan(fMRI_STAI{:,2});
IAPS_LookBase_all_compl = get_wh_image(IAPS_LookBase_all, completeCases);

% apply grey matter mask
gray_mask = fmri_mask_image('gray_matter_mask.img');
IAPS_LookBase_all_compl = IAPS_LookBase_all_compl.apply_mask(gray_mask);

% z-score outcome
IAPS_LookBase_all_compl.Y = zscore(IAPS_LookBase_all_compl.Y);
        
% z-score voxels
IAPS_LookBase_all_compl = rescale(IAPS_LookBase_all_compl, 'zscorevoxels');

% zscore images
IAPS_LookBase_all_compl = rescale(IAPS_LookBase_all_compl, 'centerimages');

% PCA
[coeffN, scoreN] = pca(IAPS_LookBase_all_compl.dat');
IAPS_LookBase_all_compl.dat = scoreN';

% train test split
cd('../holdout-identifiers')
holdoutIndex = readtable('N_IAPS_holdoutIndex');
cd('../Subject-level-maps')
IAPS_LookBase_train = get_wh_image(IAPS_LookBase_all_compl, xor(holdoutIndex.trainIndex_bin, 0));
IAPS_LookBase_test = get_wh_image(IAPS_LookBase_all_compl, xor(holdoutIndex.testIndex_bin, 0))

% train and test on holdout set
rf_model = TreeBagger(1000, IAPS_LookBase_train.dat', IAPS_LookBase_train.Y, 'Method','regression')
pattern_exp_values = predict(rf_model, IAPS_LookBase_test.dat');
[r, p] = corr(IAPS_LookBase_test.Y, pattern_exp_values)



