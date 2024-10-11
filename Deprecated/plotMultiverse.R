wd <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(wd)

if(!require("pacman")){install.packages("pacman")}
p_load(here, R.matlab, stringr, plyr, ggplot2)



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

# set negative correlations to 0
df$cvCorr <- as.numeric(ifelse(df$cvCorr < 0, 0, df$cvCorr))

df$cvCorr <- as.numeric(df$cvCorr)

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
  
  ggplot(data = df[df$outcome == outcome & df$trainsVsFull == "train", ], aes(x = reorder(modelDescr, -cvCorr), y = cvCorr)) +
    geom_point(colour = "lightgrey") +
    geom_point(data = dfMean[dfMean$outcome == outcome & dfMean$trainsVsFull == "train", ]) +
    
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

bestTrainModels <- ddply(dfMean[dfMean$trainsVsFull == "train", ], .(outcome), function(x) x[which.max(x$cvCorr),])

write.csv(bestTrainModels, here::here("Results", "Tables", "bestTrainModels.csv"), row.names = FALSE)


