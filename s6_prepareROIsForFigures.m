% exports used ROIs as niftis for plotting

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% network maps

% load network maps
[bucknermaps, mapnames] = load_image_set('bucknerlab');

for i = 1:length(mapnames)
    write(get_wh_image(bucknermaps,i), 'fname', fullfile('PatternMasks', [mapnames{i}, '.nii']), 'overwrite')
end

% correlations between single networks and neuroticism
% all non-significant, but also all negativ. 
[RHO, PVAL] = corr(IAPS_all_compl.Y, M)

% activity in networks is largely correlated!
corr(M) > .5

triM = triu(corr(M),1)
triA = triM(triM ~= 0)
mean(triA)
[min max] = bounds(triA)

mean(triu(corr(M),1))

% multiple regression of neuroticism on network activity
fitlm([ones(size(M,1),1), M], IAPS_all_compl.Y)

[bf10 bf10approx] = bf.bfFromR2(0.0201, 332, 8)
1/bf10approx


% random forest regression
rng(668);
B = TreeBagger(1000, M, IAPS_all_compl.Y, 'Method', 'regression', 'OOBPrediction', 'On');
corr(IAPS_all_compl.Y,oobPredict(B))
%permutation test
nPerm = 1000;
outPerm = zeros(nPerm, 1);
tic
for i=1:nPerm
    
    permY = IAPS_all_compl.Y(randperm(length(IAPS_all_compl.Y)));
    BPerm = TreeBagger(1000, M, permY, 'Method', 'regression', 'OOBPrediction', 'On');
    outPerm(i) = corr(permY, oobPredict(BPerm));
    
    fprintf('Iteration #%d\n', i);
end
toc

% save permutation predictions
cd('..\..\Results\RFpermutations')
save IAPS_RFpermutation_raw.mat outPerm
cd('..\..\Data\Subject-level-maps')

% p-value for random forest prediction
cd('..\..\Results\RFpermutations')
RFperm = load('IAPS_RFpermutation_raw.mat')
cd('..\..\Data\Subject-level-maps')
sum(outPerm >= corr(IAPS_all_compl.Y,oobPredict(B)))/nPerm
histogram(outPerm)

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
[bf01, r, p] = bf.corr(IAPS_all_compl.Y,  amygdala_avrg(1,1).dat)
[bf01, r, p] = bf.corr(IAPS_all_compl.Y,  amygdala_avrg(1,2).dat)

% extract average dACC signal
dACC_avrg = extract_roi_averages(IAPS_all_compl, whole_dACC_split);
% correlate with neuroticism
[bf01, r, p] = bf.corr(IAPS_all_compl.Y,  dACC_avrg(1,1).dat)
[bf01, r, p] = bf.corr(IAPS_all_compl.Y,  dACC_avrg(1,2).dat)


% extract average aInsula signal
aIns_avrg = extract_roi_averages(IAPS_all_compl, whole_insula_split);
% correlate with neuroticism
[bf01, r, p] = bf.corr(IAPS_all_compl.Y,  aIns_avrg(1,1).dat)
[bf01, r, p] = bf.corr(IAPS_all_compl.Y,  aIns_avrg(1,2).dat)



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
IAPS_test_amy = apply_mask(IAPS_test, whole_amygdala);
[r, fullOutputs] = nestedCrossValRepeat(IAPS_train_amy, 'cv_pls', 0, maxPlsComps, 'integer', kfoldsOuter, repeats, subject_id, numWorkers);
% save
cd('../../Results/regionPLSmodels')
save IAPS_pls_amygdala_raw_nested.mat r fullOutputs
cd('../../Data/Subject-level-maps')

% evaluate
optComp_amy = optHyperpar(IAPS_train_amy, 'cv_pls', 1, maxPlsComps, 'integer', subject_id) % optimal comps: 72
[cverr, stats, optional_outputs] = predict(IAPS_train_amy, 'cv_pls', 'numcomponents', 72)
[pattern_exp_values] = apply_mask(IAPS_test_amy, stats.weight_obj, 'pattern_expression', 'ignore_missing');
[bf01, r, p] = bf.corr(IAPS_test_amy.Y, pattern_exp_values)
sum(stats.weight_obj.dat > 0 )/size(stats.weight_obj.dat, 1)

% predict from voxel-wise z-score amygdala
IAPS_train_amy_z = rescale(IAPS_train_amy, 'zscorevoxels');
IAPS_train_amy_z.Y = zscore(IAPS_train_amy.Y);
[r, fullOutputs] = nestedCrossValRepeat(IAPS_train_amy_z, 'cv_pls', 0, maxPlsComps, 'integer', kfoldsOuter, repeats, subject_id, numWorkers);
% save
cd('../../Results/regionPLSmodels')
save IAPS_pls_amygdala_z_nested.mat r fullOutputs
cd('../../Data/Subject-level-maps')


% predict from image-wise z-score amygdala
IAPS_train_amy_z2 = rescale(IAPS_train_amy_z, 'zscoreimages');
[r, fullOutputs] = nestedCrossValRepeat(IAPS_train_amy_z2, 'cv_pls', 0, maxPlsComps, 'integer', kfoldsOuter, repeats, subject_id, numWorkers);
% save
cd('../../Results/regionPLSmodels')
save IAPS_pls_amygdala_z2_nested.mat r fullOutputs
cd('../../Data/Subject-level-maps')


%%%%%%%%%%%%%
% dACC

% predict from raw dACC
IAPS_train_dACC = apply_mask(IAPS_train, whole_dACC);
IAPS_test_dACC = apply_mask(IAPS_test, whole_dACC);
[r, fullOutputs] = nestedCrossValRepeat(IAPS_train_dACC, 'cv_pls', 0, maxPlsComps, 'integer', kfoldsOuter, repeats, subject_id, numWorkers);
% save
cd('../../Results/regionPLSmodels')
save IAPS_pls_dACC_raw_nested.mat r fullOutputs
cd('../../Data/Subject-level-maps')

% evaluate
optComp_dACC = optHyperpar(IAPS_train_dACC, 'cv_pls', 1, maxPlsComps, 'integer', subject_id) % optimal comps: 84
[cverr, stats, optional_outputs] = predict(IAPS_train_dACC, 'cv_pls', 'numcomponents', 84)
[pattern_exp_values] = apply_mask(IAPS_test_dACC, stats.weight_obj, 'pattern_expression', 'ignore_missing');
[bf01, r, p] = bf.corr(IAPS_test_dACC.Y, pattern_exp_values)
sum(stats.weight_obj.dat > 0 )/size(stats.weight_obj.dat, 1)


% predict from voxel-wise z-score dACC
IAPS_train_dACC_z = rescale(IAPS_train_dACC, 'zscorevoxels');
IAPS_train_dACC_z.Y = zscore(IAPS_train_dACC.Y);
[r, fullOutputs] = nestedCrossValRepeat(IAPS_train_dACC_z, 'cv_pls', 0, maxPlsComps, 'integer', kfoldsOuter, repeats, subject_id, numWorkers);
% save
cd('../../Results/regionPLSmodels')
save IAPS_pls_dACC_z_nested.mat r fullOutputs
cd('../../Data/Subject-level-maps')

% predict from image-wise z-score dACC
IAPS_train_dACC_z2 = rescale(IAPS_train_dACC_z, 'zscoreimages');
[r, fullOutputs] = nestedCrossValRepeat(IAPS_train_dACC_z2, 'cv_pls', 0, maxPlsComps, 'integer', kfoldsOuter, repeats, subject_id, numWorkers);
% save
cd('../../Results/regionPLSmodels')
save IAPS_pls_dACC_z2_nested.mat r fullOutputs
cd('../../Data/Subject-level-maps')


%%%%%%%%%%%%%
% aIns

% predict from raw aIns
IAPS_train_aIns = apply_mask(IAPS_train, whole_insula);
IAPS_test_aIns = apply_mask(IAPS_test, whole_insula);
[r, fullOutputs] = nestedCrossValRepeat(IAPS_train_aIns, 'cv_pls', 0, maxPlsComps, 'integer', kfoldsOuter, repeats, subject_id, numWorkers);
% save
cd('../../Results/regionPLSmodels')
save IAPS_pls_aIns_raw_nested.mat r fullOutputs
cd('../../Data/Subject-level-maps')

% evaluate
optComp_aIns = optHyperpar(IAPS_train_aIns, 'cv_pls', 1, maxPlsComps, 'integer', subject_id) % optimal comps: 84
[cverr, stats, optional_outputs] = predict(IAPS_train_aIns, 'cv_pls', 'numcomponents', 84)
[pattern_exp_values] = apply_mask(IAPS_test_aIns, stats.weight_obj, 'pattern_expression', 'ignore_missing');
[bf01, r, p] = bf.corr(IAPS_test_aIns.Y, pattern_exp_values)
sum(stats.weight_obj.dat > 0 )/size(stats.weight_obj.dat, 1)


% predict from voxel-wise z-score aIns
IAPS_train_aIns_z = rescale(IAPS_train_aIns, 'zscorevoxels');
IAPS_train_aIns_z.Y = zscore(IAPS_train_aIns.Y);
[r, fullOutputs] = nestedCrossValRepeat(IAPS_train_aIns_z, 'cv_pls', 0, maxPlsComps, 'integer', kfoldsOuter, repeats, subject_id, numWorkers);
% save
cd('../../Results/regionPLSmodels')
save IAPS_pls_aIns_z_nested.mat r fullOutputs
cd('../../Data/Subject-level-maps')

% predict from image-wise z-score aIns
IAPS_train_aIns_z2 = rescale(IAPS_train_aIns_z, 'zscoreimages');
[r, fullOutputs] = nestedCrossValRepeat(IAPS_train_aIns_z2, 'cv_pls', 0, maxPlsComps, 'integer', kfoldsOuter, repeats, subject_id, numWorkers);
% save
cd('../../Results/regionPLSmodels')
save IAPS_pls_aIns_z2_nested.mat r fullOutputs
cd('../../Data/Subject-level-maps')



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
for i=1:size(wholeBrainROIActivity,2)
    
   corrOutTrain(i) = corr(IAPS_half_train.Y, wholeBrainROIActivity(1,i).dat);
   
end

% analysis  of best region
bestRegionInd = (corrOutTrain == max(abs(corrOutTrain)));
atlas_obj.label_descriptions(bestRegionInd)
corrOutTrain(bestRegionInd)
wholeBrainROIActivityTest = extract_roi_averages(IAPS_half_test, atlas_obj);
[bf01, r, p] = bf.corr(IAPS_half_test.Y, wholeBrainROIActivityTest(1,bestRegionInd).dat)
