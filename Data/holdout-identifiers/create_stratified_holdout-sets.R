if (!require("pacman")) install.packages("pacman")
pacman::p_load(caret)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# IAPS neuroticism data
N_IAPS <- read.table("N_IAPS.txt")
set.seed(666)
testIndex <- createDataPartition(N_IAPS$V1, p = 100/nrow(N_IAPS),
                                  list = F, times = 1)
N_IAPS$V1[testIndex] <- 1
N_IAPS$V1[-testIndex] <- 0
testIndex_bin <- N_IAPS$V1
trainIndex_bin <- as.numeric(N_IAPS$V1 == F)

write.csv(data.frame(trainIndex_bin, testIndex_bin), "N_IAPS_holdoutIndex.csv", row.names = F)


# FACES neuroticism data
N_FACES <- read.table("N_FACES.txt")
set.seed(667)
testIndex <- createDataPartition(N_FACES$V1, p = 100/nrow(N_FACES),
                                 list = F, times = 1)
N_FACES$V1[testIndex] <- 1
N_FACES$V1[-testIndex] <- 0
testIndex_bin <- N_FACES$V1
trainIndex_bin <- as.numeric(N_FACES$V1 == F)

write.csv(data.frame(trainIndex_bin, testIndex_bin), "N_FACES_holdoutIndex.csv", row.names = F)

