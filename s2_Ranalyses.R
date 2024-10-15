# Evaluates and plots the results from the machine learning analyses.

# Also performs follow-up analyses on the final pattern and theory-driven analyses. If analyses are to be conducted from scratch,
# scripts 3-5 have to be ran first. If not calculated from scratch, this script uses files from the Results folder and can be run right away 


wd <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(wd)

if(!require("pacman")){install.packages("pacman")}
p_load(here, R.matlab, stringr, plyr, ggplot2, caret, BayesFactor, party, cowplot, sjstats, pwr, readxl, grid, psychometric, devEMF, lme4, svglite)


#####################################
# helper functions

# checks whether file exists before saving [csv (default) or rds allowed]
checkNsave <- function(object, outputPath, extension = "csv"){
  
  if(!file.exists(outputPath)){
    
    if(extension == "rds"){
      saveRDS(object, outputPath)
    }else if (extension == "csv"){
      write.csv(object, outputPath, row.names = FALSE)
    }
    
  }else{
    warning("File already exists in folder")
  }
  
}



#####################################
# read data

filenames <- list.files(here::here(wd, "Results", "Multiverse_Loop"))

# number of repeats and folds
exampleFile <- readMat(here::here(wd, "Results", "Multiverse_Loop", filenames[1]))
nRepeats <- length(exampleFile$fullOutputs)
nFolds <- exampleFile$fullOutputs[[1]][[3]]

# loop to load data into df
df <- as.data.frame(matrix(nrow = 0, ncol = 11))

for(i in 1:length(filenames)){
  
  # prepare filename
  filename <- filenames[i]
  filename_noExt <- substr(filename, 1, nchar(filename)-4)
  if(substr(filename_noExt, 1, 11) == "ER_LookDiff"){filename_noExt <- str_replace(filename_noExt, "ER_LookDiff", "ERLookDiff")}
  filename_split <- str_split(filename_noExt, "_", simplify = T)
  
  # load file
  modelResults <- readMat(here::here(wd, "Results", "Multiverse_Loop", filename))
  
  for(j in 1:nRepeats){
    
    resultRows <- cbind(matrix(filename_split, nrow=nFolds, ncol=length(filename_split), byrow=TRUE), modelResults$fullOutputs[[j]][[5]], filename_noExt)
    df <- rbind(df, resultRows)
    
  }
  
}


#####################################
# preprocess data

names(df) <- c("outcome", "data", "contrast", "trainsVsFull", "masking", "rescale", "cv", "algorithm", "cvCorr", "optHyperPar", "modelDescr")

df$cvCorr <- as.numeric(df$cvCorr)
df$data <- ifelse(df$data == "Faces-PFA", "PFA", df$data)

# calculate mean correlation per model (including fisher z transformation)
df <- ddply(df, c("outcome", "data", "contrast", "trainsVsFull", "masking", "rescale", "cv", "algorithm"), transform, meanCorr = tanh(mean(atanh(cvCorr))))



#####################################
# plot data


################
# multiverse plots

hist(df$cvCorr)

dfMean <- unique(df[, -which(names(df) == "cvCorr" | names(df) == "optHyperPar")])
names(dfMean)[which(names(dfMean) == "meanCorr")] <- "cvCorr"


plotMultiverse <- function(outcome){
  
  ggplot(data = df[df$outcome == outcome, ], aes(x = reorder(modelDescr, -cvCorr), y = cvCorr)) +
    geom_point(colour = "lightgrey") +
    geom_point(data = dfMean[dfMean$outcome == outcome, ]) +
    
    geom_hline(yintercept = mean(df[df$outcome == outcome, "cvCorr"]), linetype = "dashed") +
    
    theme_classic() +
    theme(axis.text.x = element_text(angle = 90, vjust = 1, hjust=1)) +
    
    ggtitle(outcome) + xlab("Model") + ylab("Pattern-Outcome Correlation")
  
}

plotMultiverse("ERLookDiff")
plotMultiverse("BDI")
plotMultiverse("neoN")
plotMultiverse("NA")
plotMultiverse("PA")
plotMultiverse("STAI")
plotMultiverse("neoN1")
plotMultiverse("neoN2")
plotMultiverse("neoN6")



################
# make list of best models per construct based on training data

dfMean$trainsVsFull <- ifelse(dfMean$outcome == "NEONother" | dfMean$outcome == "NEONX", "AHAB", dfMean$trainsVsFull)

# first stage neuroticism model set
dfMean[dfMean$outcome == "neoN" & dfMean$algorithm == "pls" & dfMean$trainsVsFull == "train", ]
ddply(dfMean[dfMean$outcome == "neoN" & dfMean$algorithm == "pls" & dfMean$trainsVsFull == "train", ], .(outcome), function(x) x[which.max(x$cvCorr),])

# second stage neuroticism model set
df_2ndstage <-  dfMean[dfMean$trainsVsFull != "full" & 
         dfMean$algorithm != "rf" & 
         dfMean$outcome %in% c("neoN", "neoN1", "neoN2", "neoN3", "neoN4", "neoN5", "neoN6"), ]
df_2ndstage[df_2ndstage$cvCorr == max(df_2ndstage$cvCorr), ]

ddply(dfMean[dfMean$trainsVsFull != "full" & 
               dfMean$algorithm != "rf" & 
               dfMean$outcome %in% c("neoN", "neoN1", "neoN2", "neoN3", "neoN4", "neoN5", "neoN6"), ], .(outcome), function(x) x[which.max(x$cvCorr),])


# all models
ddply(dfMean, .(outcome), function(x) x[which.max(x$cvCorr),])

if(!file.exists(here::here("Results", "Tables", "bestTrainModels.csv"))){
  write.csv(ddply(dfMean[dfMean$trainsVsFull != "full", ], .(outcome), function(x) x[which.max(x$cvCorr),]), 
            here::here("Results", "Tables", "bestTrainModels.csv"),
            row.names = FALSE)
}else{
  warning("File already exists in folder")
}



#################
# plot by dataset and aspects
#NEON1	Anxiety NEON2	Angry Hostility NEON3	Depression NEON4	Self-Consciousness NEON5	Impulsiveness  NEON6	Vulnerability 


df_boxplotFacettes <- df_2ndstage[df_2ndstage$outcome != "neoN", ]
df_boxplotFacettes$outcome <- factor(df_boxplotFacettes$outcome, c("neoN2", "neoN5", "neoN1", "neoN3", "neoN4", "neoN6"))


plotFac <- ggplot(data = df_boxplotFacettes, aes(y = cvCorr, x = outcome, fill = data)) +
  
  geom_boxplot() +
  scale_x_discrete(labels = c("Angry Hostility", "Impulsiveness", "Anxiety", "Depression", "Self-Consciousness", "Vulnerability")) +
  
  theme_classic() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_discrete(labels = c("scenes", "faces")) +
  guides(fill = guide_legend(title = "Task")) +
  theme(legend.position="top") +
  
  labs(x = NULL, y = "Correlations (cross-validated)")
  
  
ggsave(here::here("Figures", "Boxplot_FacettesData.png"), device = "png")
ggsave(here::here("Figures", "Boxplot_FacettesData.pdf"), device = "pdf")



scales::show_col(scales::hue_pal()(2))

plotOther <- ggplot(data = dfMean[dfMean$outcome %in% c("neoN", "NEONX", "NEONother") & dfMean$trainsVsFull != "full" & dfMean$algorithm != "rf", ], 
       aes(y = cvCorr, x = outcome)) +
  geom_boxplot(fill = "grey") +
  theme_classic() +
  xlab(NULL) + ylab("Correlations (cross-validated)") +
  scale_x_discrete(labels = c("Self-Report", "Other-Report", "Combined")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), axis.title.y = element_text(colour="white")) +
  theme(plot.margin = unit(c(0.6, 0, 0.3, 0), 
                           "inches"))

ggsave(here::here("Figures", "Boxplot_selfOther.png"), device = "png")
ggsave(here::here("Figures", "Boxplot_selfOther.pdf"), device = "pdf")

plot_grid(plotFac, plotOther, labels = c('a', 'b'))
ggsave(here::here("Figures", "Boxplots_FacetsselfOther.pdf"), device = "pdf", width = 6.75, height = 3.75, units = "in")
ggsave(here::here("Figures", "Boxplots_FacetsselfOther.png"), device = "png", width = 6.75, height = 3.75, units = "in")
ggsave(here::here("Figures", "Boxplots_FacetsselfOther.svg"), device = "svg", width = 6.75, height = 3.75, units = "in")



#################
# line plot of best models MORE TICK MARKS. MAKE PANEL PLOT

# Major constructs
df_others_sort <- dfMean[dfMean$outcome %in% c("neoN", "NEONother", "NA", "PA", "STAI", "BDI", "ERLookDiff"), ]
df_others_sort <- arrange(df_others_sort, outcome, desc(cvCorr))
df_others_sort <- ddply(df_others_sort, "outcome", transform, modelRank = seq_along(outcome))
df_others_sort$outcome <- factor(df_others_sort$outcome, levels = c("neoN", "NEONother", "NA", "PA", "STAI", "BDI", "ERLookDiff"))

facetLabs <- c("Neuroticism: Self", "Neuroticism: Other", "Negative Affect", "Positive Affect", "STAI", "BDI", "Task Ratings")

p1 <- ggplot(data = df_others_sort, aes(x = modelRank, y = cvCorr, colour = outcome)) +
  
  geom_point(size = 0.7) +
  geom_line() +
  
  theme_classic() +
  
  theme(axis.text.x=element_blank(),axis.ticks.x=element_blank(),axis.title.x=element_blank(), axis.line.x=element_blank()) + 
  theme(legend.position = c(0.7, 0.83), legend.text = element_text(size = 6), legend.spacing.y = unit(0.01, "cm")) +
  guides(colour = guide_legend(title = NULL)) +
  scale_color_discrete(labels = facetLabs) +
  geom_hline(yintercept=0) +
  scale_y_continuous(breaks=round(seq(-0.3, 0.4, 0.1),1), limits=c(-0.3,0.4)) +
  theme(axis.title.y = element_text(colour = "white")) +
  theme(legend.key.spacing.y = unit(-0.2, "cm"), legend.box.spacing = unit(-0.2, "cm"))



# neuroticism and facettes
df_facettes_sort <- dfMean[dfMean$outcome %in% c("neoN1", "neoN2", "neoN3", "neoN4", "neoN5", "neoN6"), ]
df_facettes_sort <- arrange(df_facettes_sort, outcome, desc(cvCorr))
df_facettes_sort$modelRank <- rep(c(1:96), 6)
#df_facettes_sort$outcome <- factor(df_facettes_sort$outcome, levels = c(Angry Hostility", "Impulsiveness", "Anxiety", "Depression", "Self-Consciousness", "Vulnerability"))
df_facettes_sort$outcome <- factor(df_facettes_sort$outcome, levels = unique(df_facettes_sort$outcome))

facetLabs <- c("Angry Hostility", "Impulsiveness", "Anxiety", "Depression", "Self-Consciousness", "Vulnerability")

p2 <- ggplot(data = df_facettes_sort, aes(x = modelRank, y = cvCorr, colour = outcome)) +
  
  geom_point(size = 0.7) +
  geom_line() +
  
  theme_classic() +
  
  theme(axis.text.x=element_blank(),axis.ticks.x=element_blank(),axis.title.x=element_blank(), axis.line.x=element_blank()) + 
  theme(legend.position = c(0.7, 0.85), legend.text = element_text(size = 6)) +
  guides(colour = guide_legend(title = NULL)) +
  scale_color_discrete(labels = facetLabs) +
  geom_hline(yintercept=0) +
  scale_y_continuous(breaks=round(seq(-0.3, 0.4, 0.1),1), limits = c(-0.3, 0.4)) +
  ylab("Correlations (cross-validated)") +
  theme(legend.key.spacing.y = unit(-0.2, "cm"), legend.box.spacing = unit(-0.2, "cm"))
  




plot_grid(p2, p1, labels = c('a','b'))


ggsave(here::here("Figures", "lineplotFacettes.emf"), device = {function(filename, ...) devEMF::emf(file = filename, ...)}, width = 6.75, height = 4, units = "in")
ggsave(here::here("Figures", "lineplotFacettes.png"), device = "png", width = 6.75, height = 4, units = "in")
ggsave(here::here("Figures", "lineplotFacettes.pdf"), device = "pdf", width = 6.75, height = 4, units = "in")




################
# Bayes factors

bestSecondStage <- read.csv(here::here("Results", "holdOutModels", "BestSecondStageModelData.csv"))
cor.test(bestSecondStage[,1], bestSecondStage[,2], alternative = "greater")
cor.test(bestSecondStage[,1], bestSecondStage[,2])
correlationBF(bestSecondStage[,1], bestSecondStage[,2], nullInterval = c(0, 1))
correlationBF(bestSecondStage[,1], bestSecondStage[,2], nullInterval = c(0, 1), rscale="medium.narrow")



################
# random forest on design factors


names(df)
df_design <- df[, -c(5, 7, 10, 11, 12)]
df_design$contrast <- ifelse(df_design$contrast == "LookNeg-vs-Baseline" | df_design$contrast == "Faces-vs-Baseline", "implBaseline", "controlCond")
df_design$trainsVsFull <- ifelse(df_design$outcome == "NEONother" | df_design$outcome == "NEONX", "AHAB", df_design$trainsVsFull)
df_design[,1:6] <- data.frame(unclass(df_design[,1:6]), stringsAsFactors = TRUE)
head(df_design)
write.csv(df_design, here::here(wd, "NDesignShinyApp", "dfDesign.csv"), row.names = FALSE)

#rf_model <- cforest(data = df_design, cvCorr ~ ., controls = cforest_unbiased(mtry = 6, ntree = 1000))
checkNsave(rf_model, here::here(wd, "Results", "designFactors","RFmodel_designfactors.rds"), extension = "rds")
rf_model <- readRDS(here::here(wd, "Results", "designFactors","RFmodel_designfactors.rds"))

#predictedValues <- predict(rf_model, OOB = TRUE)
checkNsave(predictedValues, here::here(wd, "Results", "designFactors", "predictedValues.csv"))
predictedValues <- read.csv(here::here(wd, "Results", "designFactors", "predictedValues.csv"))

R2 <- 1 - (sum((df_design$cvCorr - predictedValues)^2)/sum((df_design$cvCorr - mean(df_design$cvCorr))^2))
R2

#varImp <- varImp(rf_model, conditional = TRUE)
#dfvarImp <- data.frame(names = row.names(varImp), imp = varImp$Overall)
# checkNsave(dfvarImp, here::here(wd, "Results", "designFactors", "varImp.csv"))

dfvarImp <- read.csv(here::here(wd, "Results", "designFactors", "varImp.csv"))
# according to https://www.ibe.med.uni-muenchen.de/organisation/mitarbeiter/070_drittmittel/janitza/rf_ordinal/tutorial.pdf ,
# varimp of the cforest package computes MSE as the importance measure, which is not as interpretable as RMSE.
dfvarImp$imp <- sqrt(dfvarImp$imp) 

# LME4 model
lmeDesign_model <- lmer(data=df_design, cvCorr ~ 1 + (1|outcome) + (1|data) + (1|contrast) + (1|trainsVsFull) + (1|rescale) + (1|algorithm))
lmeDesign <- as.data.frame(VarCorr(lmeDesign_model))[,c("grp","sdcor")]


# plot results
designComb <- rbind(dfvarImp, setNames(lmeDesign, names(dfvarImp)))
designComb <- designComb[1:(nrow(designComb)-1), ]
designComb$analysis <- rep(c("Variable Importance", "Variance Decomposition"), each=6)
designComb$names <- factor(designComb$names, levels=c("outcome","data", "algorithm", "contrast", "trainsVsFull", "rescale"))


# Define custom fill colors
custom_colors <- c('Variable Importance' = '#00BFC4', 'Variance Decomposition' = '#F87660')  # Adjust these values based on your actual Type values

# Create the plot
ggplot(data = designComb, aes(x = names, y = imp, fill = analysis)) +
  geom_bar(stat='identity', position="dodge") +
  
  ylab("Correlation Metric") +
  xlab(NULL) +
  
  scale_fill_manual(values = custom_colors) +             # Manually set fill colors
  
  theme_classic() +
  
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),    # Rotate x-axis labels
    legend.title = element_blank(),                       # Remove legend label
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    axis.line = element_line(colour = "black",),
    legend.position = c(0.7, 0.8),
    legend.text = element_text(size = 7),  # Adjust text size
    legend.key.size = unit(0.4, "cm")      # Adjust key size
  ) +
  scale_y_continuous(expand = c(0, 0), breaks=seq(0, 0.2, 0.025), limits=c(0,0.2)) +# Ensure x-axis crosses y-axis at zero
  scale_x_discrete(labels = c("Construct", "Task", "Algorithm", "Baseline", "Full Data", "Scaling"))





ggsave(here::here("Figures", "varImpVarComp.png"), device = "png", width = 3, height = 3, units = "in")


#######################################################################################################################
# plot theory-driven results


dfTheo <- read_excel(here::here(wd, "Results", "Tables", "theoryDrivenResults.xlsx"))


########
# region plot

# prepare df
dfregion <- dfTheo[dfTheo$Level == "region",-c(1, 5, 8)]
dfregion <- data.frame(Type=c("Average Left", "Average Right", "Average Left", "Average Right", "Average Left", "Average Right", "Pattern", "Pattern", "Pattern", "Cerebellum/Diencephalon"),
                       dfregion)
dfregion <- cbind(reshape2::melt(dfregion[,c("Type", "Test", "r_IAPS", "r_FACES")], id.vars = c("Test", "Type")), 
                  reshape2::melt(dfregion[,c("Type", "Test", "C_IAPS", "C_FACES")], id.vars = c("Test", "Type")))
dfregion[,8] <- str_replace_all(dfregion[,8], "\\[|\\]", "")
dfregion <- cbind(dfregion, str_split_fixed(dfregion[,8], ",", n=2))
dfregion <- dfregion[,-c(5,6,7,8)]
names(dfregion)[3:6] <- c("task", "correlation", "lower", "upper")
dfregion$task <- ifelse(str_detect(dfregion$task, "FACES"), "FACES", "IAPS")
dfregion$Type <- ifelse(dfregion$Type == "Cerebellum/Diencephalon" & dfregion$task == "IAPS",
                        "Cerebellum",
                        ifelse(dfregion$Type == "Cerebellum/Diencephalon" & dfregion$task == "FACES",
                               "Diencephalon",
                               dfregion$Type))
dfregion <- dfregion[order(dfregion$Test), ]
dfregion$region <- c(rep("aInsula", 6), rep("Amygdala", 6), rep("Best Region", 2), rep("dACC", 6))
dfregion$correlation <- as.numeric(dfregion$correlation)
dfregion$lower <- as.numeric(dfregion$lower)
dfregion$upper <- as.numeric(dfregion$upper)
dfregion$Type <- str_replace_all(dfregion$Type, "Average ", "")
dfregion$Type  <- factor(dfregion$Type, levels=c("Left", "Right", "Pattern", "Cerebellum", "Diencephalon"))
dfregion$region <- factor(dfregion$region, levels = c("Amygdala", "aInsula", "dACC", "Best Region"))
dfregion$task <- factor(dfregion$task, levels=c("IAPS", "FACES"))

dfregion <- rbind(dfregion, dfregion[13:14, ])
dfregion[21:22,5:6] <- NA
dfregion$trainTest <- c(rep("test", 20), "train", "train")
dfregion[21,4] <- 0.23
dfregion[22,4] <- 0.27
dfregion$trainTest <- factor(dfregion$trainTest, levels=c("test","train"))

dfregionA <- dfregion[1:20,]
dfregionB <- dfregion[21:22,]

# create plot

regPlot <- ggplot(data=dfregionA, aes(x=Type, y=correlation, colour=task)) +
  geom_hline(yintercept = 0, linetype="dashed", colour = "lightgrey") +
  geom_point(size = 2, position=position_dodge(width=0.6)) +
  geom_errorbar(aes(ymax=upper, ymin=lower), position=position_dodge(width=0.6), width=0.4) +
  geom_point(data=dfregionB, aes(x=Type, y=correlation, colour=task, fill=trainTest), shape=1, show.legend = FALSE) +
  facet_wrap(~region, scales = "free_x", strip.position = "bottom", nrow=1) +
  
  scale_y_continuous(breaks=seq(-0.5, 0.5, 0.1), limits=c(-0.5,0.5), labels=c("", "-0.4", "", "-0.2", "", "0", "", "0.2", "", "0.4", "")) +
  xlab(NULL) + ylab("Correlation") + labs(colour=NULL) +
  ggtitle("Regions") +
  
  theme_classic() +
  theme(strip.placement = "outside") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1)) +
  theme(axis.title.y = element_text(size = 14, color = "white")) +
  theme(legend.position = c(0.85, 1.05), legend.direction = "horizontal") +
  theme(plot.title = element_text(hjust = 0.5, face = "bold")) +
  theme(strip.background = element_blank(), panel.border = element_blank(), panel.spacing = unit(0.5, "lines")) +
  theme(strip.placement = "outside", strip.text.x = element_text(vjust = -0.5)) +
  theme(strip.text = element_text(face = "italic")) + 
  scale_colour_discrete(labels = c("scenes", "faces")) 

regPlot

ggsave(here::here(wd, "Figures", "regionFigure.pdf"), width = 6.75, height = 3.25, units = "in")



########
# networks

# prepare df
dfnetwork <- dfTheo[dfTheo$Level == "network",-c(1, 5, 8)]
dfnetwork <- cbind(reshape2::melt(dfnetwork[,c("Test", "r_IAPS", "r_FACES")], id.vars = c("Test")), 
                  reshape2::melt(dfnetwork[,c("Test", "C_IAPS", "C_FACES")], id.vars = c("Test")))
dfnetwork <- dfnetwork[,-c(4,5)]
dfnetwork[,4] <- str_replace_all(dfnetwork[,4], "\\[|\\]", "")
dfnetwork <- cbind(dfnetwork, str_split_fixed(dfnetwork[,4], ",", n=2))
dfnetwork <- dfnetwork[,-4]
names(dfnetwork)[4:5] <- c("lower", "upper")
dfnetwork$lower <- ifelse(dfnetwork$Test == "linearRegress" | dfnetwork$Test == "randomForest", NA, dfnetwork$lower)
dfnetwork$value <- as.numeric(dfnetwork$value)
dfnetwork$lower <- as.numeric(dfnetwork$lower)
dfnetwork$upper <- as.numeric(dfnetwork$upper)
dfnetwork$task <- ifelse(str_detect(dfnetwork$variable, "FACES"), "FACES", "IAPS")
dfnetwork$Type <- rep(c("Visual", "Somatomotor", "Dorsal Attention", "Ventral Attention", "Limbic", "Fronto-Parietal", "Default-Mode", "Multiple Regression", "Random Forest"),2)
dfnetwork$Type <- factor(dfnetwork$Type, levels = c("Limbic",  "Ventral Attention", "Dorsal Attention", "Fronto-Parietal", "Default-Mode", "Visual", "Somatomotor", "Multiple Regression", "Random Forest"))
dfnetwork$task <- factor(dfnetwork$task, levels=c("IAPS", "FACES"))


# create plot
netPlot <- ggplot(data=dfnetwork, aes(x=Type, y=value, colour=task)) +
  geom_hline(yintercept = 0, linetype="dashed", colour = "lightgrey") +
  geom_point(size = 2, position=position_dodge(width=0.6)) +
  geom_errorbar(aes(ymax=upper, ymin=lower), position=position_dodge(width=0.6), width=0.4) +
  annotate(geom="text", x=8, y=0.25, label="n.s.", colour="darkgrey") +
  annotate(geom="text", x=9, y=0.25, label="n.s.", colour="darkgrey") +
  
  scale_y_continuous(breaks=seq(-0.5, 0.5, 0.1), limits=c(-0.5,0.5), labels=c("", "-0.4", "", "-0.2", "", "0", "", "0.2", "", "0.4", "")) +
  xlab(NULL) + ylab("Product-Moment Correlation") + labs(colour=NULL) +
  ggtitle("Networks") +
  
  theme_classic() +
  theme(strip.placement = "outside") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1)) +
  theme(axis.title.y = element_text(size = 14, color = "black")) +
  #theme(legend.position = c(0.5, 0.9), legend.direction = "horizontal") +
  theme(plot.title = element_text(hjust = 0.5, face = "bold")) +
  theme(legend.position="none")

  ggsave(here::here(wd, "Figures", "networksFigure.pdf"), width = 6.75, height = 3.25, units = "in")
  
  
  
  ########
  # signatures
  
  # prepare df
  dfsign <- dfTheo[dfTheo$Level == "signature",-c(1, 5, 8)]
  dfsign <- cbind(reshape2::melt(dfsign[,c("Test", "r_IAPS", "r_FACES")], id.vars = c("Test")), 
                     reshape2::melt(dfsign[,c("Test", "C_IAPS", "C_FACES")], id.vars = c("Test")))
  dfsign <- dfsign[,-c(4,5)]
  dfsign[,4] <- str_replace_all(dfsign[,4], "\\[|\\]", "")
  dfsign <- cbind(dfsign, str_split_fixed(dfsign[,4], ",", n=2))
  dfsign <- dfsign[,-4]
  names(dfsign)[4:5] <- c("lower", "upper")
  dfsign$value <- as.numeric(dfsign$value)
  dfsign$lower <- as.numeric(dfsign$lower)
  dfsign$upper <- as.numeric(dfsign$upper)
  dfsign$task <- ifelse(str_detect(dfsign$variable, "FACES"), "FACES", "IAPS")
  dfsign$task <- factor(dfsign$task, levels=c("IAPS", "FACES"))
  dfsign$Test <- factor(dfsign$Test, levels=c("PINES", "VIFS", "Fear", "Anger", "Sadness"))
  
  # plot signatures
  sigPlot <- ggplot(data=dfsign, aes(x=Test, y=value, colour=task)) +
    geom_hline(yintercept = 0, linetype="dashed", colour = "lightgrey") +
    geom_point(size = 2, position=position_dodge(width=0.6)) +
    geom_errorbar(aes(ymax=upper, ymin=lower), position=position_dodge(width=0.6), width=0.4) +

    scale_y_continuous(breaks=seq(-0.5, 0.5, 0.1), limits=c(-0.5,0.5), labels=c("", "-0.4", "", "-0.2", "", "0", "", "0.2", "", "0.4", "")) +
    xlab(NULL) + ylab("Correlation") + labs(colour=NULL) +
    ggtitle("Signatures") +
    
    theme_classic() +
    theme(strip.placement = "outside") +
    theme(axis.text.x = element_text(angle = 30, hjust = 1)) +
    theme(axis.title.y = element_text(size = 14, color = "white")) +
    #theme(legend.position = c(0.5, 0.9), legend.direction = "horizontal") +
    theme(plot.title = element_text(hjust = 0.5, face = "bold")) +
    theme(legend.position="none")
  sigPlot
  
  ggsave(here::here(wd, "Figures", "signaturesFigure.pdf"), width = 6.75, height = 3.25, units = "in")
  

  ########
  # combine plots
  
  plot_grid(regPlot, netPlot, sigPlot, 
            ncol=1, 
            rel_heights = c(1,1,0.8), 
            labels = c('a', 'b', 'c'), hjust = -3.3, vjust = 2, label_size = 12)
  
  ggsave(here::here(wd, "Figures", "theoryDrivenFigure.pdf"), width = 6.75, height = 6.75, units = "in")
  ggsave(here::here(wd, "Figures", "theoryDrivenFigure.svg"), width = 6.75, height = 6.75, units = "in")
  
  
  
 
  
  
  #######################################################################################################################
  # Plot lesioned accuracies
  
  # load and prepare data
  lesDat <- read.csv(here::here(wd, "Results", "holdOutModels", "lesionResults.csv")) # N = 102
  lesDat <- rbind(c("Full Model", 0.19, 0.028), c("Sparse Mask", 0.20, 0.04), lesDat)
  lesDat$r <- as.numeric(lesDat$r)
  lesDat$p <- as.numeric(lesDat$p)
  lesDat$Type <- factor(c("Full Model", "Sparse Mask", rep("Networks",7), rep("Regions", 3)), levels=c("Full Model", "Sparse Mask", "Regions", "Networks"))
  lesDat$Region <- factor(lesDat$Region, levels=c("Full Model", "Sparse Mask", lesDat$Region[10:12], lesDat$Region[3:9]))
  
  # sapply(lesDat$r, function(x){CIr(x,n=102)})
  

  
  # Define custom fill colors
custom_colors <- c('Full Model' = 'black', 'Sparse Mask' = "darkgrey", 'Regions' = '#F87660', 'Networks' = '#00BFC4')  # Adjust these values based on your actual Type values
  
  # Create the plot
ggplot(data = lesDat, aes(x = Region, y = r, fill = Type)) +
    geom_col() +
    
    ylab("Correlation") +
    xlab(NULL) +
    
    scale_fill_manual(values = custom_colors) +             # Manually set fill colors
    
    theme_classic() +
    
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),    # Rotate x-axis labels
      legend.title = element_blank(),                       # Remove legend label
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_blank(),
      axis.line = element_line(colour = "black",),
      legend.position = c(0.85, 0.85)
    ) +
  scale_y_continuous(expand = c(0, 0), breaks=seq(0, 0.3, 0.025), limits=c(0,0.3)) + # Ensure x-axis crosses y-axis at zero
  theme(legend.key.size = unit(0.5, "cm"),
        legend.text = element_text(size = 8))

ggsave(here::here("Figures", "Lesions.png"), width = 3.75, height = 3.75, units = "in")
ggsave(here::here("Figures", "Lesions.svg"), width = 3.75, height = 3.75, units = "in")


  
