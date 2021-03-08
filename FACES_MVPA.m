%%%%%%%% Analyses on FACES paradigm %%%%%%%% 

% dependencies: spm, canlabcoretools, canlab_single_trials
% put "Function" folder on path
% go to Data directory 'subject-level-maps' as the base directory

% settings
numWorkers = 5; % for parallel computing
kfoldsOuter = 5;
repeats = 2;

% example image for the paradigm of interest
betaName = 'Subject_001_Faces-PFA_Faces-vs-Shapes.nii.gz';

% locate image
myfile = which(betaName);
mydir = fileparts(myfile);
if isempty(mydir), disp('Uh-oh! I can''t find the data.'), else disp('Data found.'), end

% load FACES files into fmri_data object
image_names = filenames(fullfile(mydir, '*Faces-PFA_Faces-vs-Shapes.nii.gz'), 'absolute');
FACES_all = fmri_data(image_names);

% check descriptives/outliers
% descriptives(FACES_all);
% plot(FACES_all)
[ds, expectedds, p, wh_outlier_uncorr, wh_outlier_corr] = mahal(FACES_all, 'noplot', 'corr'); 

% one bonferroni-corrected outlier. next higher p-value not an outlier
% according to both bonferroni and bonferroni-holm
BH_threshold = transpose((0.05./sort([1:length(p)], 'descend')));
p_sort = sort(p);
[p_sort BH_threshold]

FACES_all = get_wh_image(FACES_all, ~wh_outlier_corr);

% make list of unpadded fMRI IDs
[P, N, E] = cellfun(@fileparts, image_names(~wh_outlier_corr), 'UniformOutput', false);
id_fMRI = extractBetween(N, 9, 11);
id_fMRI = str2double(id_fMRI);
id_fMRI = array2table(id_fMRI, 'VariableNames', {'id'});
FACES_all.metadata_table.subject_id = id_fMRI;

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
FACES_all.Y = fMRI_NEON.NEON;
completeCases = ~isnan(fMRI_NEON.NEON);
FACES_all_compl = get_wh_image(FACES_all, completeCases);

% mask grey matter
gray_mask = fmri_mask_image('gray_matter_mask.img');
FACES_all_compl = FACES_all_compl.apply_mask(gray_mask);

% create stratified hold-out set 
%(not implemented in matlab for specific hold-out N and continuous vars)
cd('../holdout-identifiers')
writematrix(FACES_all_compl.Y, 'N_FACES')
holdoutIndex = readtable('N_FACES_holdoutIndex');
cd('../Subject-level-maps')
FACES_train = get_wh_image(FACES_all_compl, xor(holdoutIndex.trainIndex_bin, 0));
FACES_test = get_wh_image(FACES_all_compl, xor(holdoutIndex.testIndex_bin, 0));

% test prediction [maybe important: predict appears to center the outcome]
[cverr, stats, optional_outputs] = predict(FACES_train, 'algorithm_name', 'cv_pcr', 'numcomponents', floor(sum(holdoutIndex{:,1})*(4/5)-1), 'nfolds', 5);
%[pattern_exp_values] = apply_mask(FACES_train, stats.weight_obj, 'pattern_expression', 'ignore_missing')
% corr(FACES_train.Y, pattern_exp_values)

% get subject ID from fMRI_data training set
[~,~,subject_id] = unique(FACES_train.metadata_table.subject_id,'stable');
maxPlsComps = floor(length(subject_id)*(4/5)^2)-1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% predict on raw data

% predict
[r, fullOutputs] = nestedCrossValRepeat(FACES_train, 'cv_pls', 0, maxPlsComps, 'integer', kfoldsOuter, repeats, subject_id, numWorkers);

% save
cd('../../Results')
save FACES_pls_raw_nested.mat r fullOutputs
cd('../Data/Subject-level-maps')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% predict on cross-sectionally z-scored data

% standardize IVs and DVs cross-sectionally
FACES_train_z = rescale(FACES_train, 'zscorevoxels');
FACES_train_z.Y = zscore(FACES_train_z.Y);

% predict
[r, fullOutputs] = nestedCrossValRepeat(FACES_train_z, 'cv_pls', 0, maxPlsComps, 'integer', kfoldsOuter, repeats, subject_id, numWorkers);

% save
cd('../../Results')
save FACES_pls_z_nested.mat r fullOutputs
cd('../Data/Subject-level-maps')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% predict on cross-sectionally and image-wise z-scored data

% image-wise standardization
FACES_train_z2 = rescale(FACES_train_z, 'zscoreimages');

% predict
[r, fullOutputs] = nestedCrossValRepeat(FACES_train_z2, 'cv_pls', 0, maxPlsComps, 'integer', kfoldsOuter, repeats, subject_id, numWorkers);

% save
cd('../../Results')
save FACES_pls_z2_nested.mat r fullOutputs
cd('../Data/Subject-level-maps')




% %%%%%%%%%%%%%%%%%%%%%
% % test code
% 
% fmri_data_file = which('bmrk3_6levels_pain_dataset.mat');
% 
% if isempty(fmri_data_file)
% 
%     % attempt to download
%     disp('Did not find data locally...downloading data file from figshare.com')
% 
%     fmri_data_file = websave('bmrk3_6levels_pain_dataset.mat', 'https://ndownloader.figshare.com/files/12708989');
% 
% end
% 
% load(fmri_data_file);
% 
% % subject_id is useful for cross-validation
% subject_id_pain = image_obj.additional_info.subject_id;
% 
% % ratings: reconstruct a subjects x temperatures matrix
% ratings = reshape(image_obj.Y, 6, 33)';
% 
% % temperatures are 44 - 49 (actually 44.3 - 49.3) in order for each person.
% temperatures = image_obj.additional_info.temperatures;
% 
% % use function
% [r, r_ind, hyp] = nestedCrossVal(image_obj, 'cv_pcr', 149, 150, 'integer', 2, subject_id)




