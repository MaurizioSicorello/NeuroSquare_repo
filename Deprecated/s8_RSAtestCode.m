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


% calculate dissimilarity matrix for (a) neuroticism scores and (b) images

% neuroticism (euclidean distance)
N_dis = pdist(IAPS_all_compl.Y);
N_dis_square = squareform(N_dis);

% images (cosine similarity)
image_dis = pdist(IAPS_all_compl.dat', 'cosine');
image_dis_square = squareform(image_dis);

% rank correlation between both dissimilarity matrices
triu_idx = logical(triu(ones(size(N_dis_square)), 1));
corr(N_dis_square(triu_idx), image_dis_square(triu_idx), 'Type', 'Spearman')

