%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Alternative approaches to predict neuroticism scores from brain data
% (neural signatures, networks, and regions)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load data

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

%store Neuroticism in fmri_object and subset complete cases
IAPS_all.Y = fMRI_NEON.NEON;
completeCases = ~isnan(fMRI_NEON.NEON);
IAPS_all_compl = get_wh_image(IAPS_all, completeCases);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% neural signature approach

% load neural signatures
cd('..\..\PatternMasks')
pines = fmri_data('Rating_Weights_LOSO_2.nii');
fearKragel = fmri_data('Kragelfearful.nii');
angerKragel = fmri_data('Kragelangry.nii');
sadnessKragel = fmri_data('Kragelsad.nii');
fearFeng = fmri_data('VIFS.nii');
cd('..\Data\Subject-level-maps')

% dot product
[RHO, PVAL] = corr(IAPS_all_compl.Y, apply_mask(IAPS_all_compl, pines, 'pattern_expression', 'ignore_missing'))
[RHO, PVAL] = corr(IAPS_all_compl.Y, apply_mask(IAPS_all_compl, fearKragel, 'pattern_expression', 'ignore_missing'))
[RHO, PVAL] = corr(IAPS_all_compl.Y, apply_mask(IAPS_all_compl, angerKragel, 'pattern_expression', 'ignore_missing'))
[RHO, PVAL] = corr(IAPS_all_compl.Y, apply_mask(IAPS_all_compl, sadnessKragel, 'pattern_expression', 'ignore_missing'))
[RHO, PVAL] = corr(IAPS_all_compl.Y, apply_mask(IAPS_all_compl, fearFeng, 'pattern_expression', 'ignore_missing'))

% check correlation matrix of Kragel patterns for CFA 
patternExpressionMatrix = [apply_mask(IAPS_all_compl, fearKragel, 'pattern_expression', 'ignore_missing'), ...
    apply_mask(IAPS_all_compl, angerKragel, 'pattern_expression', 'ignore_missing'), ...
    apply_mask(IAPS_all_compl, sadnessKragel, 'pattern_expression', 'ignore_missing')];
corr(patternExpressionMatrix) % low or strongly negative correlations. CFA doesn't make sense

% cosine similarity
[RHO, PVAL] = corr(IAPS_all_compl.Y, apply_mask(IAPS_all_compl, pines, 'pattern_expression', 'ignore_missing', 'cosine_similarity'))
[RHO, PVAL] = corr(IAPS_all_compl.Y, apply_mask(IAPS_all_compl, fearKragel, 'pattern_expression', 'ignore_missing', 'cosine_similarity'))
[RHO, PVAL] = corr(IAPS_all_compl.Y, apply_mask(IAPS_all_compl, angerKragel, 'pattern_expression', 'ignore_missing', 'cosine_similarity'))
[RHO, PVAL] = corr(IAPS_all_compl.Y, apply_mask(IAPS_all_compl, sadnessKragel, 'pattern_expression', 'ignore_missing', 'cosine_similarity'))
[RHO, PVAL] = corr(IAPS_all_compl.Y, apply_mask(IAPS_all_compl, fearFeng, 'pattern_expression', 'ignore_missing', 'cosine_similarity'))

% correlation similarity
[RHO, PVAL] = corr(IAPS_all_compl.Y, apply_mask(IAPS_all_compl, pines, 'pattern_expression', 'ignore_missing', 'correlation'))
[RHO, PVAL] = corr(IAPS_all_compl.Y, apply_mask(IAPS_all_compl, fearKragel, 'pattern_expression', 'ignore_missing', 'correlation'))
[RHO, PVAL] = corr(IAPS_all_compl.Y, apply_mask(IAPS_all_compl, angerKragel, 'pattern_expression', 'ignore_missing', 'correlation'))
[RHO, PVAL] = corr(IAPS_all_compl.Y, apply_mask(IAPS_all_compl, sadnessKragel, 'pattern_expression', 'ignore_missing', 'correlation'))
[RHO, PVAL] = corr(IAPS_all_compl.Y, apply_mask(IAPS_all_compl, fearFeng, 'pattern_expression', 'ignore_missing', 'correlation'))


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% network approach

% load network maps
[bucknermaps, mapnames] = load_image_set('bucknerlab');

% create matrix with activity within networks
M = zeros(length(IAPS_all_compl.Y), length(mapnames));
for i = 1:length(mapnames)
    M(:,i) = extract_roi_averages(IAPS_all_compl, bucknermaps.get_wh_image(i)).dat;
end

% correlations between single networks and neuroticism
% all non-significant, but also all negativ. 
[RHO, PVAL] = corr(IAPS_all_compl.Y, M)

% activity in networks is largely correlated!
corr(M)

% multiple regression of neuroticism on network activity
[~,~,~,~,stats] = regress(IAPS_all_compl.Y, [ones(size(M,1),1), M])

% random forest regression
rng(668);
B = TreeBagger(1000, M, IAPS_all_compl.Y, 'Method', 'regression', 'OOBPrediction', 'On');
corr(IAPS_all_compl.Y,oobPredict(B))
%permutation test
nPerm = 100;
outPerm = zeros(nPerm, 1);
for i=1:nPerm
    
    permY = IAPS_all_compl.Y(randperm(length(IAPS_all_compl.Y)));
    BPerm = TreeBagger(1000, M, permY, 'Method', 'regression', 'OOBPrediction', 'On');
    outPerm(i) = corr(permY, oobPredict(BPerm));
    
    fprintf('Iteration #%d\n', i);
end

% save permutation predictions
cd('..\..\Results')
save IAPS_RFpermutation_raw.mat nPerm
cd('..\Data\Subject-level-maps')

% p-value for random forest prediction
sum(outPerm >= corr(IAPS_all_compl.Y,oobPredict(B)))/nPerm


%%%%%%%%%%%%%%%%%%%%%%%%%
% repeat linear analyses on brain-wise centered fMRI data
IAPS_all_compl_cen = rescale(IAPS_all_compl, 'centerimages');

M = zeros(length(IAPS_all_compl_cen.Y), length(mapnames));
for i = 1:length(mapnames)
    M(:,i) = extract_roi_averages(IAPS_all_compl_cen, bucknermaps.get_wh_image(i)).dat;
end

% correlations between single networks and neuroticism
% all non-significant, but also all negativ. 
[RHO, PVAL] = corr(IAPS_all_compl_cen.Y, M)

% correlation beween activity in different networks is lower now!
corr(M)

% multiple regression of neuroticism on network activity
[~,~,~,~,stats] = regress(IAPS_all_compl_cen.Y, [ones(size(M,1),1), M])

% random forest regression
rng(668);
B = TreeBagger(1000, M, IAPS_all_compl_cen.Y, 'Method', 'regression', 'OOBPrediction', 'On');
corr(IAPS_all_compl_cen.Y,oobPredict(B))




%%%%%%%%%%%%%%%%%%%%%%%%%
% repeat linear analyses on brain-wise z-scored fMRI data
IAPS_all_compl_z2 = rescale(IAPS_all_compl, 'zscoreimages');

M = zeros(length(IAPS_all_compl_z2.Y), length(mapnames));
for i = 1:length(mapnames)
    M(:,i) = extract_roi_averages(IAPS_all_compl_z2, bucknermaps.get_wh_image(i)).dat;
end

% correlations between single networks and neuroticism
% actually, one is significant...
[RHO, PVAL] = corr(IAPS_all_compl_z2.Y, M)
mapnames(PVAL<=0.05/7)

% correlation beween activity in different networks is lower now!
corr(M)

% multiple regression of neuroticism on network activity
[~,~,~,~,stats] = regress(IAPS_all_compl_z2.Y, [ones(size(M,1),1), M])

% random forest regression
rng(668);
B = TreeBagger(1000, M, IAPS_all_compl_z2.Y, 'Method', 'regression', 'OOBPrediction', 'On');
corr(IAPS_all_compl_z2.Y,oobPredict(B))



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% region approach


%%%%%%%%%%%%%%%%%%%%%%%%%%%
% prepare masks

% load atlas
atlas_obj = load_atlas('canlab2018_2mm');

% make amygdala masks
% [encompassing all amygdala regions in the atlas]
whole_amygdala = select_atlas_subset(atlas_obj, {'Amygdala'}, 'flatten');
orthviews(whole_amygdala) % the spm toolbox amygdala ROI always struck me as very large
whole_amygdala_split = split_atlas_by_hemisphere(whole_amygdala);

% make dACC masks
% [encompassing all regions labelled as former BA32 in Glasser 2016]
dACC = select_atlas_subset(atlas_obj, {'Ctx_p32pr', 'Ctx_d32', 'Ctx_p32','Ctx_a32pr'});
orthviews(dACC) % p32 is detached from the remaining subregions (already visible in Glasser (2016), Fig. S22)
% make ROI without this subregion
whole_dACC = select_atlas_subset(atlas_obj, {'Ctx_p32pr', 'Ctx_d32', 'Ctx_a32pr'}, 'flatten');
whole_dACC_split = split_atlas_by_hemisphere(whole_dACC);

% make insula masks
% anterior and middle insular regions from Glasser (2016) [maybe overinclusive?] 
whole_insula = select_atlas_subset(atlas_obj, {'Ctx_MI_', 'Ctx_AVI','Ctx_AAIC'}, 'flatten');
orthviews(whole_insula)
whole_insula_split = split_atlas_by_hemisphere(whole_insula);


%%%%%%%%%%%%%%%%%%%%%%%%%%
% average signal within regions

% extract average amygdala signal
amygdala_avrg = extract_roi_averages(IAPS_all_compl, whole_amygdala_split);
% correlate with neuroticism
corr(IAPS_all_compl.Y, amygdala_avrg(1,1).dat) % left hemisphere
corr(IAPS_all_compl.Y, amygdala_avrg(1,2).dat) % right hemisphere

% extract average dACC signal
dACC_avrg = extract_roi_averages(IAPS_all_compl, whole_dACC_split);
% correlate with neuroticism
corr(IAPS_all_compl.Y, dACC_avrg(1,1).dat) % left hemisphere
corr(IAPS_all_compl.Y, dACC_avrg(1,2).dat) % right hemisphere

% extract average aInsula signal
aIns_avrg = extract_roi_averages(IAPS_all_compl, whole_insula_split);
% correlate with neuroticism
corr(IAPS_all_compl.Y, aIns_avrg(1,1).dat) % left hemisphere
corr(IAPS_all_compl.Y, aIns_avrg(1,2).dat) % right hemisphere


%%%%%%%%%%%%%%%%%%%%%%%%%%
% MVPA on regions of interest

% split data in training and test
cd('../holdout-identifiers')
holdoutIndex = readtable('N_IAPS_holdoutIndex');
cd('../Subject-level-maps')
IAPS_train = get_wh_image(IAPS_all_compl, xor(holdoutIndex.trainIndex_bin, 0));
IAPS_test = get_wh_image(IAPS_all_compl, xor(holdoutIndex.testIndex_bin, 0));
[~,~,subject_id] = unique(IAPS_train.metadata_table.subject_id,'stable');

% settings
numWorkers = 5; % for parallel computing
kfoldsOuter = 5;
repeats = 2;
maxPlsComps = floor(size(IAPS_train.dat,2)*(4/5)^2)-1;


%%%%%%%%%%%%%
% Amygdala

% predict from raw amygdala
IAPS_train_amy = apply_mask(IAPS_train, whole_amygdala);
[r, fullOutputs] = nestedCrossValRepeat(IAPS_train_amy, 'cv_pls', 0, maxPlsComps, 'integer', kfoldsOuter, repeats, subject_id, numWorkers);
% save
cd('../../Results')
save IAPS_pls_amygdala_raw_nested.mat r fullOutputs
cd('../Data/Subject-level-maps')

% predict from voxel-wise z-score amygdala
IAPS_train_amy_z = rescale(IAPS_train_amy, 'zscorevoxels');
IAPS_train_amy_z.Y = zscore(IAPS_train_amy.Y);
[r, fullOutputs] = nestedCrossValRepeat(IAPS_train_amy_z, 'cv_pls', 0, maxPlsComps, 'integer', kfoldsOuter, repeats, subject_id, numWorkers);
% save
cd('../../Results')
save IAPS_pls_amygdala_z_nested.mat r fullOutputs
cd('../Data/Subject-level-maps')

% predict from image-wise z-score amygdala
IAPS_train_amy_z2 = rescale(IAPS_train_amy_z, 'zscoreimages');
[r, fullOutputs] = nestedCrossValRepeat(IAPS_train_amy_z2, 'cv_pls', 0, maxPlsComps, 'integer', kfoldsOuter, repeats, subject_id, numWorkers);
% save
cd('../../Results')
save IAPS_pls_amygdala_z2_nested.mat r fullOutputs
cd('../Data/Subject-level-maps')


%%%%%%%%%%%%%
% dACC

% predict from raw dACC
IAPS_train_dACC = apply_mask(IAPS_train, whole_dACC);
[r, fullOutputs] = nestedCrossValRepeat(IAPS_train_dACC, 'cv_pls', 0, maxPlsComps, 'integer', kfoldsOuter, repeats, subject_id, numWorkers);
% save
cd('../../Results')
save IAPS_pls_dACC_raw_nested.mat r fullOutputs
cd('../Data/Subject-level-maps')

% predict from voxel-wise z-score dACC
IAPS_train_dACC_z = rescale(IAPS_train_dACC, 'zscorevoxels');
IAPS_train_dACC_z.Y = zscore(IAPS_train_dACC.Y);
[r, fullOutputs] = nestedCrossValRepeat(IAPS_train_dACC_z, 'cv_pls', 0, maxPlsComps, 'integer', kfoldsOuter, repeats, subject_id, numWorkers);
% save
cd('../../Results')
save IAPS_pls_dACC_z_nested.mat r fullOutputs
cd('../Data/Subject-level-maps')

% predict from image-wise z-score dACC
IAPS_train_dACC_z2 = rescale(IAPS_train_dACC_z, 'zscoreimages');
[r, fullOutputs] = nestedCrossValRepeat(IAPS_train_dACC_z2, 'cv_pls', 0, maxPlsComps, 'integer', kfoldsOuter, repeats, subject_id, numWorkers);
% save
cd('../../Results')
save IAPS_pls_dACC_z2_nested.mat r fullOutputs
cd('../Data/Subject-level-maps')


%%%%%%%%%%%%%
% aIns

% predict from raw aIns
IAPS_train_aIns = apply_mask(IAPS_train, whole_insula);
[r, fullOutputs] = nestedCrossValRepeat(IAPS_train_aIns, 'cv_pls', 0, maxPlsComps, 'integer', kfoldsOuter, repeats, subject_id, numWorkers);
% save
cd('../../Results')
save IAPS_pls_aIns_raw_nested.mat r fullOutputs
cd('../Data/Subject-level-maps')

% predict from voxel-wise z-score aIns
IAPS_train_aIns_z = rescale(IAPS_train_aIns, 'zscorevoxels');
IAPS_train_aIns_z.Y = zscore(IAPS_train_aIns.Y);
[r, fullOutputs] = nestedCrossValRepeat(IAPS_train_aIns_z, 'cv_pls', 0, maxPlsComps, 'integer', kfoldsOuter, repeats, subject_id, numWorkers);
% save
cd('../../Results')
save IAPS_pls_aIns_z_nested.mat r fullOutputs
cd('../Data/Subject-level-maps')

% predict from image-wise z-score aIns
IAPS_train_aIns_z2 = rescale(IAPS_train_aIns_z, 'zscoreimages');
[r, fullOutputs] = nestedCrossValRepeat(IAPS_train_aIns_z2, 'cv_pls', 0, maxPlsComps, 'integer', kfoldsOuter, repeats, subject_id, numWorkers);
% save
cd('../../Results')
save IAPS_pls_aIns_z2_nested.mat r fullOutputs
cd('../Data/Subject-level-maps')



%%%%%%%%%%%%%%%%%%%%%%%%%%
% best-region approach

% use inbuild stratification of predict function to make train/test set
[cverr, stats, optout] = predict(IAPS_all_compl, 'nfolds', 2);
IAPS_half_train = get_wh_image(IAPS_all_compl, stats.trIdx{1,1});
IAPS_half_test = get_wh_image(IAPS_all_compl, stats.teIdx{1,1});

% get average activity for ROIs covering most of the brain
wholeBrainROIActivity = extract_roi_averages(IAPS_half_train, atlas_obj); % includes bilateral regions so far. Not explicitly defined in the prereg

% correlate with neuroticism
corrOutTrain = zeros(size(wholeBrainROIActivity,2), 1);
corrOutTest = zeros(size(wholeBrainROIActivity,2), 1);
for i=1:size(wholeBrainROIActivity,2)
    
   corrOutTrain(i) = corr(IAPS_half_train.Y, wholeBrainROIActivity(1,i).dat);
   corrOutTest(i) = corr(IAPS_half_test.Y, wholeBrainROIActivity(1,i).dat);
   
end

% correlation in test sample
atlas_obj.label_descriptions(corrOutTrain == max(abs(corrOutTrain)))
corrOutTest(corrOutTrain == max(abs(corrOutTrain)))

