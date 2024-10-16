% Computes all machine learning models reported in the paper.


%%%%%%%%%%%%%%%%%%%%%%%%%%
% manual settings
numWorkers = 3; % for parallel computing
kfoldsOuter = 5;
repeats = 2;
onlyUseTrainData = false;

% settings for testing the loop
testMode = false;
testMode_N = 100; % sample size
testMode_k = 20; % number of features


%%%%%%%%%%%%%%%%%%%%%%%%%%
% list of algorithms to use
ML_algorithm = ["cv_pcr" , "cv_svr", "oob_rf", "cv_pls"];


%%%%%%%%%%%%%%%%%%%%%%%%%%
% scaling options to try
scale_opt = ["zscoreimages", "centerimages", "nocen"];



%%%%%%%%%%%%%%%%%%%%%%%%%%
% example images of fMRI data
fMRI_datasets = ["IAPS_neg-vs-neut", "*IAPS_LookNeg-vs-LookNeut.nii"; ...
    "IAPS_neg-vs-base", "*IAPS_LookNeg-vs-Baseline.nii"; ...
    "FACES_neg-vs-neut", "*Faces-PFA_Faces-vs-Shapes.nii"; ...
     "FACES_neg-vs-base", "*PFA_Faces-vs-Baseline_mean.nii"];
 



%%%%%%%%%%%%%%%%%%%%%%%%%%
% load questionnaire data
cd('Data')
AHAB2_quest = readtable('AHAB2_psychVars_deidentified');
PIP_quest = readtable('PIP_psychVars_deidentified');

%AHAB2_vars = AHAB2_quest(:,{'id', 'NEON', 'NEON1', 'NEON2', 'NEON3', 'NEON4', 'NEON5', 'NEON6', 'ER_LookDiff', 'pnsx_pa', 'pnsx_na', 'STAI', 'BDI_TOT','NEONX'});
%PIP_vars = PIP_quest(:,{'id', 'neoN', 'neoN1', 'neoN2', 'neoN3', 'neoN4', 'neoN5', 'neoN6', 'ER_LookDiff', 'PA_rescale', 'NA_rescale', 'Trait_Anxiety', 'BDI_total', 'NEONX_empty'});

%All_Y = array2table([AHAB2_vars{:,:}; PIP_vars{:,:}], 'VariableNames', ...
%   {'id', 'neoN', 'neoN1', 'neoN2', 'neoN3', 'neoN4', 'neoN5', 'neoN6', 'ER_LookDiff', 'PA', 'NA', 'STAI', 'BDI', 'NEONX'});

AHAB2_vars = AHAB2_quest(:,{'id', 'NEONX'});
PIP_vars = PIP_quest(:,{'id', 'NEONX_empty'});

All_Y = array2table([AHAB2_vars{:,:}; PIP_vars{:,:}], 'VariableNames', ...
   {'id', 'NEONother'});



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAIN LOOP

% cd to fMRI data folder
cd('Subject-level-maps')

%%%%%%%%%%%%%%%%%%%%%%%%%%
% loop to load outcome variable
for i=2:size(All_Y, 2)
    
    % load outcome
    outcome = All_Y(:,[1,i]);
    
    % if DV is other rating, make combined score
    if string(outcome.Properties.VariableNames{2}) == 'NEONX'
        
        outcome = All_Y(:,{'id','neoN','NEONX'});
        outcome = outcome(all(~ismissing(outcome),2),:);
        outcome.neoN_both = (zscore(outcome.neoN) + zscore(outcome.NEONX))/2;
        outcome = outcome(:,[1,4]);
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    % loop to load brain dataset
    for j=1:size(fMRI_datasets, 1)
        
        % skip if looking at difference between negative/neutral pictures
        % in non-applicable datasets
        if string(outcome.Properties.VariableNames{2}) == 'ER_LookDiff' & fMRI_datasets(j,1) ~= 'IAPS_neg-vs-neut'
            continue
        end
        
        % load fMRI data
        image_names = filenames(fullfile(pwd, char(fMRI_datasets(j,2))), 'absolute');
        fMRI_all = fmri_data(image_names);
        
        % remove outliers (bonferroni-corrected)
        % [ds, expectedds, p, wh_outlier_uncorr, wh_outlier_corr] = mahal(fMRI_all, 'noplot', 'corr');
        % only one outlier present in 1 out of 4 datasets. Did not change results for main models,
        % so proceed with all data for simplicity
        
        % make list of unpadded fMRI IDs
        [P, N, E] = cellfun(@fileparts, image_names, 'UniformOutput', false);
        id_fMRI = extractBetween(N, 9, 11);
        id_fMRI = str2double(id_fMRI);
        id_fMRI = array2table(id_fMRI, 'VariableNames', {'id'});
        fMRI_all.metadata_table.subject_id = id_fMRI;
        
        % save info on dataset
        fMRI_data_info = strsplit(N{1}, '_');
        
        % join fMRI IDs with outcome data
        fMRI_outcome = innerjoin(id_fMRI, outcome);
        
        % store DV in fmri_object and subset complete cases
        fMRI_all.Y = fMRI_outcome{:,2};
        completeCases = ~isnan(fMRI_outcome{:,2});
        fMRI_all_compl = get_wh_image(fMRI_all, completeCases);
        
        % apply grey matter mask
        gray_mask = fmri_mask_image('gray_matter_mask.img');
        fMRI_all_compl = fMRI_all_compl.apply_mask(gray_mask);
        
        % option to use only training vs all data
        if onlyUseTrainData & string(outcome.Properties.VariableNames{2}) ~= 'neoN_both'
            cd('../holdout-identifiers')
            
            if j == 1 | j == 2
                holdoutIndex = readtable('N_IAPS_holdoutIndex');
            elseif j == 3 | j == 4
                holdoutIndex = readtable('N_FACES_holdoutIndex');
            end
                
            cd('../Subject-level-maps')
            trainInd = xor(holdoutIndex.trainIndex_bin, 0);
            fMRI_all_compl = get_wh_image(fMRI_all_compl, trainInd(1:min(length(fMRI_all_compl.Y), length(trainInd))));
            
            trainingOrFull = 'train';
            
        else
            trainingOrFull = 'full';
        end
        
        % z-score outcome
        fMRI_all_compl.Y = zscore(fMRI_all_compl.Y);
        
        % z-score voxels
        fMRI_all_compl = rescale(fMRI_all_compl, 'zscorevoxels');
        
        
        % if test mode, replace data with simulated data
        if testMode == true
            fMRI_all_compl.dat = zscore(normrnd(0, 1, testMode_N, testMode_k)', 0, 2);
            fMRI_all_compl.Y = zscore(zscore(fMRI_all_compl.dat'*normrnd(0, 0.5, testMode_k, 1)) + normrnd(0, testMode_k*0.1, testMode_N, 1));
        end
        
      
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%
        % loop through analysis with/without neurosynth masking
        for k=1:2
            
            
            if k==2
                % neurosynth masking code not written yet. skip for now
                continue
                %maskOpt = 'NS';
            else
                maskOpt = 'GM';
            end
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%
            % loop through scaling options
            for l=1:length(scale_opt)
                
                if l~=3
                    fMRI_all_scaled = rescale(fMRI_all_compl, scale_opt(l));   
                else
                    fMRI_all_scaled = fMRI_all_compl;
                end
                
            
                %%%%%%%%%%%%%%%%%%%%%%%%%%
                % loop through algorithms ADD SUBJECT ID CODE
                for m=1:(length(ML_algorithm))
                    
                    % create list of subject ids
                    if testMode == true
                        subject_id = [1:testMode_N]';
                    else
                        [~,~,subject_id] = unique(fMRI_all_scaled.metadata_table.subject_id,'stable');
                    end
                    
                    % output label of model
                    outLabel = strcat(All_Y.Properties.VariableNames{i}, '_', ...
                            fMRI_data_info{3}, '_', ...
                            fMRI_data_info{4}, '_', ...
                            trainingOrFull, '_', ...
                            maskOpt, '_', ...
                            extractBefore(scale_opt(l), 6), '_', ...
                            ML_algorithm(m), '.mat');
                    outLabel = char(outLabel);
                    
                    % calculate model if it doesn't already exist
                    disp(['CALCULATING THE FOLLOWING MODEL: ', outLabel])
                    cd('../../Results/Multiverse_Loop')
                    if isfile(outLabel) ~= 1
                        
                        if ML_algorithm(m) == 'oob_rf'
                        [r, fullOutputs] = cv_ranfor_repeat(fMRI_all_scaled, kfoldsOuter, repeats, 1000, numWorkers, true, subject_id)
                        
                        elseif ML_algorithm(m) == 'cv_svr'
                        [r, fullOutputs] = cv_svr_repeat(fMRI_all_scaled, kfoldsOuter, repeats, subject_id)
                        
                        else
                        maxComps = floor(length(subject_id)*((kfoldsOuter-1)/kfoldsOuter)^2)-1;
                        [r, fullOutputs] = nestedCrossValRepeat(fMRI_all_scaled, char(ML_algorithm(m)), 1, maxComps-1, 'integer', kfoldsOuter, repeats, subject_id, 0);     
                        end
                        
                        % save model
                        save(outLabel, 'r', 'fullOutputs')
                        
                    else
                        disp('MODEL ALREADY EXISTS IN RESULTS FOLDER. SKIPPING TO NEXT MODEL')
                        
                    end
                   
                    cd('../../Data/Subject-level-maps')
                    
                end
                
            end
            
        end
        
    end     
    
end
                    
