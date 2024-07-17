


%%%%%%%%%%%%%%%%%%%%%%%%%%
% load questionnaire data
cd('Data')
AHAB2_quest = readtable('AHAB2_psychVars_deidentified');
PIP_quest = readtable('PIP_psychVars_deidentified');

AHAB2_vars = AHAB2_quest(:,{'id', 'ER_LookDiff', 'pnsx_pa', 'pnsx_na', 'STAI', 'BDI_TOT', 'NEON', 'NEON1', 'NEON2', 'NEON3', 'NEON4', 'NEON5', 'NEON6', 'NEONX', 'NEOE'});
PIP_vars = PIP_quest(:,{'id', 'ER_LookDiff', 'PA_rescale', 'NA_rescale', 'Trait_Anxiety', 'BDI_total', 'neoN', 'neoN1', 'neoN2', 'neoN3', 'neoN4', 'neoN5', 'neoN6', 'NEONX_empty', 'neoE'});

All_Y = array2table([AHAB2_vars{:,:}; PIP_vars{:,:}], 'VariableNames', ...
   {'id', 'ER_LookDiff', 'PA', 'NA', 'STAI', 'BDI', 'neoN', 'neoN1', 'neoN2', 'neoN3', 'neoN4', 'neoN5', 'neoN6', 'NEONX', 'neoE'});


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Neuroticism Facet 6
% (neoN6_IAPS_LookNeg-vs-LookNeut_train_GM_cente_cv_svr)

%%%%%%%%%%%%%%%%%%%%%%%%%%
% load fMRI dataset
cd('Subject-level-maps')
image_names = filenames(fullfile(pwd, char("*IAPS_LookNeg-vs-LookNeut.nii")), 'absolute');
IAPS_NegNeut_all = fmri_data(image_names);

% load outcome
neoN6 = All_Y(:,{'id', 'neoN6'});

% make list of unpadded fMRI IDs
[P, N, E] = cellfun(@fileparts, image_names, 'UniformOutput', false);
id_fMRI = extractBetween(N, 9, 11);
id_fMRI = str2double(id_fMRI);
id_fMRI = array2table(id_fMRI, 'VariableNames', {'id'});
IAPS_NegNeut_all.metadata_table.subject_id = id_fMRI;

% join fMRI IDs with outcome data
fMRI_neoN6 = innerjoin(id_fMRI, neoN6);
        
% store DV in fmri_object and subset complete cases
IAPS_NegNeut_all.Y = fMRI_neoN6{:,'neoN6'};
completeCases = ~isnan(fMRI_neoN6{:,2});
IAPS_NegNeut_all_compl = get_wh_image(IAPS_NegNeut_all, completeCases);

% apply grey matter mask
gray_mask = fmri_mask_image('gray_matter_mask.img');
%gray_mask = fmri_mask_image('gray_matter_mask_sparse.img');
IAPS_NegNeut_all_compl = IAPS_NegNeut_all_compl.apply_mask(gray_mask);

% z-score outcome
IAPS_NegNeut_all_compl.Y = zscore(IAPS_NegNeut_all_compl.Y);

% center images
IAPS_NegNeut_all_compl = rescale(IAPS_NegNeut_all_compl, 'centerimages');

% train test split
cd('../holdout-identifiers')
holdoutIndex = readtable('N_IAPS_holdoutIndex');
cd('../Subject-level-maps')
IAPS_neoN6_train = get_wh_image(IAPS_NegNeut_all_compl, xor(holdoutIndex.trainIndex_bin, 0));
IAPS_neoN6_test = get_wh_image(IAPS_NegNeut_all_compl, xor(holdoutIndex.testIndex_bin, 0));

% train and test on holdout set
[~,~,subject_id] = unique(IAPS_neoN6_train.metadata_table.subject_id,'stable')
hyperpar = cv_svr_optPar(IAPS_neoN6_train); % 1.6667
[cverr, stats, optional_outputs] = predict(IAPS_neoN6_train, 'cv_svr', 'C', hyperpar, 'nfolds', 1)
cd('../../Results/holdOutModels')
% write(stats.weight_obj, 'fname', 'neoN6_IAPS_LookNeg-vs-LookNeut_train_GM_cente_cv_svr.nii');

neoN6pattern = fmri_data('neoN6_IAPS_LookNeg-vs-LookNeut_train_GM_cente_cv_svr.nii');
% apply pattern to data
[pattern_exp_values] = apply_mask(IAPS_neoN6_test, neoN6pattern, 'pattern_expression', 'ignore_missing');
[r, p] = corr(IAPS_neoN6_test.Y, pattern_exp_values) % r = .168, p = .0922 (two-sided), N = 102

% writetable(table(IAPS_neoN6_test.Y, pattern_exp_values, 'VariableNames', {'TrueValues', 'Predictions'}), 'BestSecondStageModelData.csv');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% construct validity

% pattern expression for each participant using LOOCV

% Initialize an empty array
nCases = length(IAPS_NegNeut_all_compl.Y);
patternValues_LOOCV = zeros(1, nCases);


% Loop to populate the array
for i = 1:nCases
    
    fprintf('\n ITERATION: %d of 337 \n', i)
    
    % train-test split
    LOO_ind = zeros(1, nCases);
    LOO_ind(i) = 1;
    train_data = get_wh_image(IAPS_NegNeut_all_compl, xor(LOO_ind, 1));
    LO_data = get_wh_image(IAPS_NegNeut_all_compl, xor(LOO_ind, 0));
    
    % model building and prediction
    [cverr, stats, optional_outputs] = predict(train_data, 'cv_svr', 'C', 1.6667, 'nfolds', 1)
    patternValues_LOOCV(i) = LO_data.dat' * stats.weight_obj.dat;
end


dfpatternExpr = [IAPS_NegNeut_all_compl.metadata_table.subject_id, table(patternValues_LOOCV')]
writetable(dfpatternExpr, 'patternExpression_LOOCV.csv');
dfpatternExpr = readtable('patternExpression_LOOCV.csv');
mergedTable = outerjoin(All_Y, dfpatternExpr, 'Type', 'left', 'Keys', 'id');

mergedTable = innerjoin(All_Y, dfpatternExpr, 'Keys', 'id');

corr(mergedTable.Var1, mergedTable.neoN6, 'rows','complete')
corr(mergedTable.Var1, mergedTable.NA, 'rows','complete')
corr(mergedTable.Var1, mergedTable.PA, 'rows','complete')
corr(mergedTable.Var1, mergedTable.neoE, 'rows','complete')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pattern bootstrapping

% REPEAT WITH 10,000 DRAWS AND ONLY ON THE TRAINING DATA
% without saving bootsweights and saving results as mat file for reuse

%[cverr, stats, optional_outputs] = predict(IAPS_neoN6_train, 'cv_svr', 'nfolds', 1, 'C', 1.6667, ...,
%    'bootweights', 'bootsamples', 10000, 'useparallel', 0)
%cd('../../Results/holdOutModels')
%save('neoN6_IAPS_LookNeg-vs-LookNeut_train_GM_cente_cv_svr_boot10000.mat', 'stats')

%[cverr, stats, optional_outputs] = predict(IAPS_NegNeut_all_compl, 'cv_svr', 'nfolds', 1, 'C', 1.6667, ...,
%    'bootweights', 'bootsamples', 10000, 'useparallel', 0)
%cd('../../Results/holdOutModels')
%save('neoN6_IAPS_LookNeg-vs-LookNeut_full_GM_cente_cv_svr_boot10000.mat', 'stats')


% threshold results
trainModel = load('neoN6_IAPS_LookNeg-vs-LookNeut_train_GM_cente_cv_svr_boot10000.mat')
trainModelThreshFDR = threshold(trainModel.stats.weight_obj, .05, 'fdr')
trainModelThreshFDR = threshold(trainModel.stats.weight_obj, .10, 'fdr')
trainModelThreshUnc = threshold(trainModel.stats.weight_obj, .001, 'unc')

% show list of regions
reg = region(trainModelThreshFDR);
tab = table(reg);
[~, ~, regTabOut] = table(reg)

regTabOut.Sign = regTabOut.maxZ < 0
regTabOut = sortrows(regTabOut, {'Sign', 'modal_label_descriptions', 'Volume'})
regTabOut.Atlas_regions_covered = [];
regTabOut.region_index = [];
regTabOut.Sign = [];

cd('..\Tables\')
writetable(regTabOut, 'regionTable.xlsx');
cd('..\holdOutModels\')

% plot
orthviews(trainModelThreshFDR)

create_figure('montage'); axis off;
montage(trainModelThreshFDR);
drawnow, snapnow

gray_mask_tight = fmri_mask_image('gray_matter_mask_sparse.img');
threshMask = apply_mask(trainModelThreshFDR, trainModelThreshFDR)
surface(threshMask.apply_mask(gray_mask_tight)) % returns an error
orthviews(threshMask.apply_mask(gray_mask_tight))

reg_lab = autolabel_regions_using_atlas(reg);
montage(reg_lab, 'regioncenters', 'colormap');
drawnow, snapnow





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Neurosynth

neurosynthMask = fmri_data('neoN6_IAPS_LookNeg-vs-LookNeut_train_GM_cente_cv_svr.nii')

[image_by_feature_correlations, top_feature_tables] = neurosynth_feature_labels(neurosynthMask, 'images_are_replicates', false, 'noverbose');

lowwords = [top_feature_tables{1}.words_low(:)]';
disp(lowwords)

highwords = [top_feature_tables{1}.words_high(:)]';
disp(highwords)

r_low = top_feature_tables{1}.testr_low;
r_high = top_feature_tables{1}.testr_high;

r_to_plot = [r_high; r_low];
textlabels = [ highwords lowwords];

create_figure('wedge_plot');

%hh = tor_wedge_plot(r_to_plot, textlabels, 'outer_circle_radius', .3, 'colors', {[1 .7 0] [.4 0 .8]}, 'nofigure');

hh = tor_wedge_plot(r_to_plot, textlabels, 'outer_circle_radius', .3, 'colors', {[1 .7 0] [.4 0 .8]}, 'bicolor', 'nofigure');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% lesion studies

%%%%%%%%%%%%%%%%%%
% networks

% load network maps
[bucknermaps, mapnames] = load_image_set('bucknerlab');

% load region maps

% loop for lesions in network maps
netLesResults = cell(length(mapnames), 3);

for i = 1:length(mapnames)
    
    IAPStrain_netLes = apply_mask(IAPS_neoN6_train, bucknermaps.get_wh_image(i), 'invert')
    IAPStest_netLes = apply_mask(IAPS_neoN6_test, bucknermaps.get_wh_image(i), 'invert')
    
    [cverr, stats, optional_outputs] = predict(IAPStrain_netLes, 'cv_svr', 'C', 1.6667, 'nfolds', 1)
    
    patternExpr = IAPStest_netLes.dat' * stats.weight_obj.dat
    [r, p] = corr(IAPStest_netLes.Y, patternExpr)
    
    
    netLesResults{i, 1} = mapnames{i}
    netLesResults{i, 2} = r
    netLesResults{i, 3} = p
    
end


%%%%%%%%%%%%%%
% regions

% load atlas
atlas_obj = load_atlas('canlab2018_2mm');

% amygdala
whole_amygdala = fmri_data(select_atlas_subset(atlas_obj, {'Amygdala'}, 'flatten'));
IAPStrain_amyLes = apply_mask(IAPS_neoN6_train, whole_amygdala, 'invert')
IAPStest_amyLes = apply_mask(IAPS_neoN6_test, whole_amygdala, 'invert')
[cverr, stats, optional_outputs] = predict(IAPStrain_amyLes, 'cv_svr', 'C', 1.6667, 'nfolds', 1)
patternExpr = IAPStest_amyLes.dat' * stats.weight_obj.dat
[rAmy, pAmy] = corr(IAPStest_amyLes.Y, patternExpr)



% dACC 
dACC = select_atlas_subset(atlas_obj, {'Ctx_p32pr', 'Ctx_d32', 'Ctx_p32','Ctx_a32pr'});
% make ROI without this subregion
whole_dACC = fmri_data(select_atlas_subset(atlas_obj, {'Ctx_p32pr', 'Ctx_d32', 'Ctx_a32pr'}, 'flatten'));
IAPStrain_dACCLes = apply_mask(IAPS_neoN6_train, whole_dACC, 'invert')
IAPStest_dACCLes = apply_mask(IAPS_neoN6_test, whole_dACC, 'invert')
[cverr, stats, optional_outputs] = predict(IAPStrain_dACCLes, 'cv_svr', 'C', 1.6667, 'nfolds', 1)
patternExpr = IAPStest_dACCLes.dat' * stats.weight_obj.dat
[rACC, pACC] = corr(IAPStest_dACCLes.Y, patternExpr)

% make insula masks
% anterior and middle insular regions from Glasser (2016) [maybe overinclusive?] 
whole_insula = fmri_data(select_atlas_subset(atlas_obj, {'Ctx_MI_', 'Ctx_AVI','Ctx_AAIC'}, 'flatten'));
IAPStrain_insLes = apply_mask(IAPS_neoN6_train, whole_insula, 'invert')
IAPStest_insLes = apply_mask(IAPS_neoN6_test, whole_insula, 'invert')
[cverr, stats, optional_outputs] = predict(IAPStrain_insLes, 'cv_svr', 'C', 1.6667, 'nfolds', 1)
patternExpr = IAPStest_insLes.dat' * stats.weight_obj.dat
[rIns, pIns] = corr(IAPStest_insLes.Y, patternExpr)


regLesResults = cell(3, 3);
regLesResults = cell2table(regLesResults, "VariableNames",{'Region', 'r', 'p'})
regLesResults.Region = {'Amygdala'; 'dACC'; 'aInsula'};
regLesResults.r = [rAmy; rACC; rIns];
regLesResults.p = [pAmy; pACC; pIns];

netTable = cell2table(netLesResults, "VariableNames",{'Region', 'r', 'p'})
lesionTable = [netTable; regLesResults]
writetable(lesionTable, 'lesionResults.csv')


%%%%%%%%%%%%%%
% possibly artifactual voxels

gray_mask_tight = fmri_mask_image('gray_matter_mask_sparse.img');
[cverr, stats, optional_outputs] = predict(IAPS_neoN6_train.apply_mask(gray_mask_tight), 'cv_svr', 'C', 1.6667, 'nfolds', 1)

patternExpr = IAPS_neoN6_test.apply_mask(gray_mask_tight).dat' * stats.weight_obj.dat
[r, p] = corr(IAPS_neoN6_test.Y, patternExpr)
