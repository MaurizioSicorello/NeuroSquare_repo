% example image for the paradigm of interest
betaName = 'Subject_004_IAPS_LookNeg-vs-LookNeut.nii.gz';

% locate image
myfile = which(betaName);
mydir = fileparts(myfile);
if isempty(mydir), disp('Uh-oh! I can''t find the data.'), else disp('Data found.'), end

% load IAPS files into fmri_data object
image_names = filenames(fullfile(mydir, '*IAPS_LookNeg-vs-LookNeut.nii.gz'), 'absolute');
IAPS_all = fmri_data(image_names);

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
IAPS_all.X = scale(fMRI_NEON.NEON, 1);
completeCases = ~isnan(fMRI_NEON.NEON);
IAPS_all_compl = get_wh_image(IAPS_all, completeCases);

out = regress(IAPS_all_compl);
% out_fdr = threshold(out.t, .01, 'unc', 'k', 50);
out_fdr = threshold(out.t, .05, 'unc');
out_fdr = select_one_image(out_fdr, 1);
orthviews(out_fdr)

o2 = montage(out_fdr, 'trans', 'full');

sum(out_fdr.sig)/length(out_fdr.sig)


% age analyses
AHAB_age = AHAB2_quest(:,{'id', 'AGE'});
AHAB_age_compl = AHAB_age(~any(ismissing(AHAB_age),2),:);

IAPS_age_compl = get_wh_image(IAPS_all, ismember(IAPS_all.metadata_table.subject_id.id, AHAB_age_compl.id));
IAPS_age_compl.Y = table2array(AHAB_age_compl(ismember(AHAB_age_compl.id, IAPS_age_compl.metadata_table.subject_id.id),2));
IAPS_age_compl.X = table2array(AHAB_age_compl(ismember(AHAB_age_compl.id, IAPS_age_compl.metadata_table.subject_id.id),2));

IAPS_age_compl.Y = zscore(IAPS_age_compl.Y);
IAPS_age_compl = rescale(IAPS_age_compl, 'zscorevoxels');
IAPS_age_compl = rescale(IAPS_age_compl, 'zscoreimages');

[cverr, stats, optional_outputs] = predict(IAPS_age_grey, 'cv_pcr', 'ncomp', 179*4/5-1)

mask = which('gray_matter_mask.img')
maskdat = fmri_data(mask, 'noverbose');
IAPS_age_grey = apply_mask(IAPS_age_compl, maskdat);


out = regress(IAPS_age_grey);
out_fdr = threshold(out.t, .001, 'unc', 'k', 20);
out_fdr = select_one_image(out_fdr, 1);
%orthviews(out_fdr)

o2 = montage(out_fdr, 'trans', 'full');

sum(out_fdr.p <= 0.05)
