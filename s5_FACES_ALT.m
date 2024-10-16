%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Theory-driven approaches to predict neuroticism scores from brain data
% (neural signatures, networks, and regions)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load data
cd('Data/Subject-level-maps')

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

%store Neuroticism in fmri_object and subset complete cases
FACES_all.Y = fMRI_NEON.NEON;
completeCases = ~isnan(fMRI_NEON.NEON);
FACES_all_compl = get_wh_image(FACES_all, completeCases);

% extract descriptives
AHAB2_quest.Properties.VariableNames{find(string(AHAB2_quest.Properties.VariableNames) == "YRS_SCH")} = 'Yr_School'
deskrTab = join(FACES_all_compl.metadata_table.subject_id, [AHAB2_quest(:,{'id', 'age','sex', 'race', 'NEON', 'Yr_School'}); PIP_quest(:,{'id', 'age','sex', 'race', 'NEON', 'Yr_School'})])

round(mean(deskrTab.age),1)
round(std(deskrTab.age),1)
round(mean(deskrTab.NEON),1)
round(std(deskrTab.NEON),1)
round(mean(deskrTab.Yr_School),1)
round(std(deskrTab.Yr_School),1)
tabulate(deskrTab.race)
tabulate(deskrTab.sex)

cd('..\')
AHAB2_corr = readtable('AHAB2_psychVars_deidentified');
PIP_corr = readtable('PIP_psychVars_deidentified');
AHAB2_corr = AHAB2_corr(:,{'ER_LookDiff', 'pnsx_pa', 'pnsx_na', 'STAI', 'BDI_TOT', 'NEON', 'NEON1', 'NEON2', 'NEON3', 'NEON4', 'NEON5', 'NEON6', 'NEONX', 'NEOE'});
PIP_corr = PIP_corr(:,{'ER_LookDiff', 'PA_rescale', 'NA_rescale', 'Trait_Anxiety', 'BDI_total', 'neoN', 'neoN1', 'neoN2', 'neoN3', 'neoN4', 'neoN5', 'neoN6', 'NEONX_empty', 'neoE'});
corrTab = array2table([AHAB2_corr{:,:}; PIP_corr{:,:}], 'VariableNames', ...
   {'ER_LookDiff', 'PA', 'NA', 'STAI', 'BDI', 'neoN', 'neoN1', 'neoN2', 'neoN3', 'neoN4', 'neoN5', 'neoN6', 'NEONX', 'neoE'});

corrTab = corrTab(:,{'neoN'; 'neoN1'; 'neoN2'; 'neoN3'; 'neoN4'; 'neoN5'; 'neoN6'; 'NEONX'; 'NA'; 'STAI'; 'BDI'; 'PA'; 'neoE'; 'ER_LookDiff'})

descM = mean(corrTab, "omitmissing")
std(corrTab{:,:}, "omitmissing")
descC = corrcoef(corrTab{:,:}, 'Rows','pairwise')

writematrix([round(descM{:,:}',1), round(std(corrTab{:,:}, "omitmissing")', 1), round(descC, 2)], '..\Results\Tables\descrConstructs.xlsx')





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
[bf01, r, p] = bf.corr(FACES_all_compl.Y, apply_mask(FACES_all_compl, pines, 'pattern_expression', 'ignore_missing'))
[bf01, r, p] = bf.corr(FACES_all_compl.Y, apply_mask(FACES_all_compl, fearFeng, 'pattern_expression', 'ignore_missing'))
[bf01, r, p] = bf.corr(FACES_all_compl.Y, apply_mask(FACES_all_compl, fearKragel, 'pattern_expression', 'ignore_missing'))
[bf01, r, p] = bf.corr(FACES_all_compl.Y, apply_mask(FACES_all_compl, angerKragel, 'pattern_expression', 'ignore_missing'))
[bf01, r, p] = bf.corr(FACES_all_compl.Y, apply_mask(FACES_all_compl, sadnessKragel, 'pattern_expression', 'ignore_missing'))

% check correlation matrix of Kragel patterns for CFA 
patternExpressionMatrix = [apply_mask(FACES_all_compl, fearKragel, 'pattern_expression', 'ignore_missing'), ...
    apply_mask(FACES_all_compl, angerKragel, 'pattern_expression', 'ignore_missing'), ...
    apply_mask(FACES_all_compl, sadnessKragel, 'pattern_expression', 'ignore_missing')];
corr(patternExpressionMatrix) % low or strongly negative correlations. CFA doesn't make sense

% cosine similarity
[RHO, PVAL] = corr(FACES_all_compl.Y, apply_mask(FACES_all_compl, pines, 'pattern_expression', 'ignore_missing', 'cosine_similarity'))
[RHO, PVAL] = corr(FACES_all_compl.Y, apply_mask(FACES_all_compl, fearKragel, 'pattern_expression', 'ignore_missing', 'cosine_similarity'))
[RHO, PVAL] = corr(FACES_all_compl.Y, apply_mask(FACES_all_compl, angerKragel, 'pattern_expression', 'ignore_missing', 'cosine_similarity'))
[RHO, PVAL] = corr(FACES_all_compl.Y, apply_mask(FACES_all_compl, sadnessKragel, 'pattern_expression', 'ignore_missing', 'cosine_similarity'))
[RHO, PVAL] = corr(FACES_all_compl.Y, apply_mask(FACES_all_compl, fearFeng, 'pattern_expression', 'ignore_missing', 'cosine_similarity'))

% correlation similarity
[RHO, PVAL] = corr(FACES_all_compl.Y, apply_mask(FACES_all_compl, pines, 'pattern_expression', 'ignore_missing', 'correlation'))
[RHO, PVAL] = corr(FACES_all_compl.Y, apply_mask(FACES_all_compl, fearKragel, 'pattern_expression', 'ignore_missing', 'correlation'))
[RHO, PVAL] = corr(FACES_all_compl.Y, apply_mask(FACES_all_compl, angerKragel, 'pattern_expression', 'ignore_missing', 'correlation'))
[RHO, PVAL] = corr(FACES_all_compl.Y, apply_mask(FACES_all_compl, sadnessKragel, 'pattern_expression', 'ignore_missing', 'correlation'))
[RHO, PVAL] = corr(FACES_all_compl.Y, apply_mask(FACES_all_compl, fearFeng, 'pattern_expression', 'ignore_missing', 'correlation'))


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% network approach

% load network maps
[bucknermaps, mapnames] = load_image_set('bucknerlab');

% create matrix with activity within networks
M = zeros(length(FACES_all_compl.Y), length(mapnames));
for i = 1:length(mapnames)
    M(:,i) = extract_roi_averages(FACES_all_compl, bucknermaps.get_wh_image(i)).dat;
    mapnames(i)
    [bf01, r, p] = bf.corr(FACES_all_compl.Y, M(:,i))
end

% correlations between single networks and neuroticism
% all non-significant, but also all negativ. 
[RHO, PVAL] = corr(FACES_all_compl.Y, M)

% activity in networks is largely correlated!
corrM = corr(M)
mean(corrM(triu(corrM, 1) > 0))

triM = triu(corr(M),1)
triA = triM(triM ~= 0)
mean(triA)
[min max] = bounds(triA)

% multiple regression of neuroticism on network activity
fitlm([ones(size(M,1),1), M], FACES_all_compl.Y)

[bf10 bf10approx] = bf.bfFromR2(0.0307, 424, 8)
1/bf10approx

% random forest regression
rng(669);
B = TreeBagger(1000, M, FACES_all_compl.Y, 'Method', 'regression', 'OOBPrediction', 'On');
corr(FACES_all_compl.Y,oobPredict(B))
%permutation test
nPerm = 100;
outPerm = zeros(nPerm, 1);
for i=1:nPerm
    
    permY = FACES_all_compl.Y(randperm(length(FACES_all_compl.Y)));
    BPerm = TreeBagger(1000, M, permY, 'Method', 'regression', 'OOBPrediction', 'On');
    outPerm(i) = corr(permY, oobPredict(BPerm));
    
    fprintf('Iteration #%d\n', i);
end

% save permutation predictions
cd('..\..\Results\RFpermutations')
save FACES_RFpermutation_raw.mat outPerm
cd('..\..\Data\Subject-level-maps')

% p-value for random forest prediction
cd('..\..\Results\RFpermutations')
load('FACES_RFpermutation_raw.mat')
cd('..\..\Data\Subject-level-maps')
sum(outPerm >= corr(FACES_all_compl.Y,oobPredict(B)))/nPerm


%%%%%%%%%%%%%%%%%%%%%%%%%
% repeat linear analyses on brain-wise centered fMRI data
FACES_all_compl_cen = rescale(FACES_all_compl, 'centerimages');

M = zeros(length(FACES_all_compl_cen.Y), length(mapnames));
for i = 1:length(mapnames)
    M(:,i) = extract_roi_averages(FACES_all_compl_cen, bucknermaps.get_wh_image(i)).dat;
end

% correlations between single networks and neuroticism
% all non-significant, but also all negativ. 
[RHO, PVAL] = corr(FACES_all_compl_cen.Y, M)

% correlation beween activity in different networks is lower now!
corr(M)

% multiple regression of neuroticism on network activity
[~,~,~,~,stats] = regress(FACES_all_compl_cen.Y, [ones(size(M,1),1), M])

% random forest regression
rng(669);
B = TreeBagger(1000, M, FACES_all_compl_cen.Y, 'Method', 'regression', 'OOBPrediction', 'On');
corr(FACES_all_compl_cen.Y,oobPredict(B))




%%%%%%%%%%%%%%%%%%%%%%%%%
% repeat linear analyses on brain-wise z-scored fMRI data
FACES_all_compl_z2 = rescale(FACES_all_compl, 'zscoreimages');

M = zeros(length(FACES_all_compl_z2.Y), length(mapnames));
for i = 1:length(mapnames)
    M(:,i) = extract_roi_averages(FACES_all_compl_z2, bucknermaps.get_wh_image(i)).dat;
end

% correlations between single networks and neuroticism
% actually, one is significant...
[RHO, PVAL] = corr(FACES_all_compl_z2.Y, M)
mapnames(PVAL<=0.05/7)

% correlation beween activity in different networks is lower now!
corr(M)

% multiple regression of neuroticism on network activity
[~,~,~,~,stats] = regress(FACES_all_compl_z2.Y, [ones(size(M,1),1), M])

% random forest regression
rng(669);
B = TreeBagger(1000, M, FACES_all_compl_z2.Y, 'Method', 'regression', 'OOBPrediction', 'On');
corr(FACES_all_compl_z2.Y,oobPredict(B))



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
amygdala_avrg = extract_roi_averages(FACES_all_compl, whole_amygdala_split);
% correlate with neuroticism
[bf01, r, p] = bf.corr(FACES_all_compl.Y,  amygdala_avrg(1,1).dat)
[bf01, r, p] = bf.corr(FACES_all_compl.Y,  amygdala_avrg(1,2).dat)

% extract average dACC signal
dACC_avrg = extract_roi_averages(FACES_all_compl, whole_dACC_split);
% correlate with neuroticism
[bf01, r, p] = bf.corr(FACES_all_compl.Y,  dACC_avrg(1,1).dat)
[bf01, r, p] = bf.corr(FACES_all_compl.Y,  dACC_avrg(1,2).dat)


% extract average aInsula signal
aIns_avrg = extract_roi_averages(FACES_all_compl, whole_insula_split);
% correlate with neuroticism
[bf01, r, p] = bf.corr(FACES_all_compl.Y,  aIns_avrg(1,1).dat)
[bf01, r, p] = bf.corr(FACES_all_compl.Y,  aIns_avrg(1,2).dat)


%%%%%%%%%%%%%%%%%%%%%%%%%%
% MVPA on regions of interest

% split data in training and test
cd('../holdout-identifiers')
holdoutIndex = readtable('N_FACES_holdoutIndex');
cd('../Subject-level-maps')
FACES_train = get_wh_image(FACES_all_compl, xor(holdoutIndex.trainIndex_bin, 0));
FACES_test = get_wh_image(FACES_all_compl, xor(holdoutIndex.testIndex_bin, 0));
[~,~,subject_id] = unique(FACES_train.metadata_table.subject_id,'stable');

% settings
numWorkers = 5; % for parallel computing
kfoldsOuter = 5;
repeats = 2;
maxPlsComps = floor(size(FACES_train.dat,2)*(4/5)^2)-1;


%%%%%%%%%%%%%
% Amygdala

% predict from raw amygdala
FACES_train_amy = apply_mask(FACES_train, whole_amygdala);
FACES_test_amy = apply_mask(FACES_test, whole_amygdala);
[r, fullOutputs] = nestedCrossValRepeat(FACES_train_amy, 'cv_pls', 0, maxPlsComps, 'integer', kfoldsOuter, repeats, subject_id, numWorkers);
% save
cd('../../Results')
save FACES_pls_amygdala_raw_nested.mat r fullOutputs
cd('../Data/Subject-level-maps')

% evaluate
optComp_amy = optHyperpar(FACES_train_amy, 'cv_pls', 1, maxPlsComps, 'integer', subject_id) % optimal comps: 18
[cverr, stats, optional_outputs] = predict(FACES_train_amy, 'cv_pls', 'numcomponents', 18)
[pattern_exp_values] = apply_mask(FACES_test_amy, stats.weight_obj, 'pattern_expression', 'ignore_missing');
[bf01, r, p] = bf.corr(FACES_test_amy.Y, pattern_exp_values)
sum(stats.weight_obj.dat > 0 )/size(stats.weight_obj.dat, 1)


% predict from voxel-wise z-score amygdala
FACES_train_amy_z = rescale(FACES_train_amy, 'zscorevoxels');
FACES_train_amy_z.Y = zscore(FACES_train_amy.Y);
[r, fullOutputs] = nestedCrossValRepeat(FACES_train_amy_z, 'cv_pls', 0, maxPlsComps, 'integer', kfoldsOuter, repeats, subject_id, numWorkers);
% save
cd('../../Results')
save FACES_pls_amygdala_z_nested.mat r fullOutputs
cd('../Data/Subject-level-maps')

% predict from image-wise z-score amygdala
FACES_train_amy_z2 = rescale(FACES_train_amy_z, 'zscoreimages');
[r, fullOutputs] = nestedCrossValRepeat(FACES_train_amy_z2, 'cv_pls', 0, maxPlsComps, 'integer', kfoldsOuter, repeats, subject_id, numWorkers);
% save
cd('../../Results')
save FACES_pls_amygdala_z2_nested.mat r fullOutputs
cd('../Data/Subject-level-maps')


%%%%%%%%%%%%%
% dACC

% predict from raw dACC
FACES_train_dACC = apply_mask(FACES_train, whole_dACC);
FACES_test_dACC = apply_mask(FACES_test, whole_dACC);
[r, fullOutputs] = nestedCrossValRepeat(FACES_train_dACC, 'cv_pls', 0, maxPlsComps, 'integer', kfoldsOuter, repeats, subject_id, numWorkers);
% save
cd('../../Results')
save FACES_pls_dACC_raw_nested.mat r fullOutputs
cd('../Data/Subject-level-maps')

% evaluate
optComp_dACC = optHyperpar(FACES_train_dACC, 'cv_pls', 1, maxPlsComps, 'integer', subject_id) % optimal comps: 205
[cverr, stats, optional_outputs] = predict(FACES_train_dACC, 'cv_pls', 'numcomponents', 205)
[pattern_exp_values] = apply_mask(FACES_test_dACC, stats.weight_obj, 'pattern_expression', 'ignore_missing');
[bf01, r, p] = bf.corr(FACES_test_dACC.Y, pattern_exp_values)
sum(stats.weight_obj.dat > 0 )/size(stats.weight_obj.dat, 1)


% predict from voxel-wise z-score dACC
FACES_train_dACC_z = rescale(FACES_train_dACC, 'zscorevoxels');
FACES_train_dACC_z.Y = zscore(FACES_train_dACC.Y);
[r, fullOutputs] = nestedCrossValRepeat(FACES_train_dACC_z, 'cv_pls', 0, maxPlsComps, 'integer', kfoldsOuter, repeats, subject_id, numWorkers);
% save
cd('../../Results')
save FACES_pls_dACC_z_nested.mat r fullOutputs
cd('../Data/Subject-level-maps')

% predict from image-wise z-score dACC
FACES_train_dACC_z2 = rescale(FACES_train_dACC_z, 'zscoreimages');
[r, fullOutputs] = nestedCrossValRepeat(FACES_train_dACC_z2, 'cv_pls', 0, maxPlsComps, 'integer', kfoldsOuter, repeats, subject_id, numWorkers);
% save
cd('../../Results')
save FACES_pls_dACC_z2_nested.mat r fullOutputs
cd('../Data/Subject-level-maps')


%%%%%%%%%%%%%
% aIns

% predict from raw aIns
FACES_train_aIns = apply_mask(FACES_train, whole_insula);
FACES_test_aIns = apply_mask(FACES_test, whole_insula);
[r, fullOutputs] = nestedCrossValRepeat(FACES_train_aIns, 'cv_pls', 0, maxPlsComps, 'integer', kfoldsOuter, repeats, subject_id, numWorkers);
% save
cd('../../Results')
save FACES_pls_aIns_raw_nested.mat r fullOutputs
cd('../Data/Subject-level-maps')

% evaluate
optComp_aIns = optHyperpar(FACES_train_aIns, 'cv_pls', 1, maxPlsComps, 'integer', subject_id) % optimal comps: 205
[cverr, stats, optional_outputs] = predict(FACES_train_aIns, 'cv_pls', 'numcomponents', 205)
[pattern_exp_values] = apply_mask(FACES_test_aIns, stats.weight_obj, 'pattern_expression', 'ignore_missing');
[bf01, r, p] = bf.corr(FACES_test_aIns.Y, pattern_exp_values)
sum(stats.weight_obj.dat > 0 )/size(stats.weight_obj.dat, 1)


% predict from voxel-wise z-score aIns
FACES_train_aIns_z = rescale(FACES_train_aIns, 'zscorevoxels');
FACES_train_aIns_z.Y = zscore(FACES_train_aIns.Y);
[r, fullOutputs] = nestedCrossValRepeat(FACES_train_aIns_z, 'cv_pls', 0, maxPlsComps, 'integer', kfoldsOuter, repeats, subject_id, numWorkers);
% save
cd('../../Results')
save FACES_pls_aIns_z_nested.mat r fullOutputs
cd('../Data/Subject-level-maps')

% predict from image-wise z-score aIns
FACES_train_aIns_z2 = rescale(FACES_train_aIns_z, 'zscoreimages');
[r, fullOutputs] = nestedCrossValRepeat(FACES_train_aIns_z2, 'cv_pls', 0, maxPlsComps, 'integer', kfoldsOuter, repeats, subject_id, numWorkers);
% save
cd('../../Results')
save FACES_pls_aIns_z2_nested.mat r fullOutputs
cd('../Data/Subject-level-maps')



%%%%%%%%%%%%%%%%%%%%%%%%%%
% best-region approach

% use inbuild stratification of predict function to make train/test set
[cverr, stats, optout] = predict(FACES_all_compl, 'nfolds', 2);
FACES_half_train = get_wh_image(FACES_all_compl, stats.trIdx{1,1});
FACES_half_test = get_wh_image(FACES_all_compl, stats.teIdx{1,1});

% get average activity for ROIs covering most of the brain
wholeBrainROIActivity = extract_roi_averages(FACES_half_train, atlas_obj); % includes bilateral regions so far. Not explicitly defined in the prereg

% correlate with neuroticism
corrOutTrain = zeros(size(wholeBrainROIActivity,2), 1);
for i=1:size(wholeBrainROIActivity,2)
    
   corrOutTrain(i) = corr(FACES_half_train.Y, wholeBrainROIActivity(1,i).dat);
   
end

% analysis of best region
bestRegionInd = (corrOutTrain == max(abs(corrOutTrain)));
atlas_obj.label_descriptions(bestRegionInd)
corrOutTrain(bestRegionInd)
wholeBrainROIActivityTest = extract_roi_averages(FACES_half_test, atlas_obj);
[bf01, r, p] = bf.corr(FACES_half_test.Y, wholeBrainROIActivityTest(1,bestRegionInd).dat)
