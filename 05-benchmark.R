source("01-data.R")
# ======================
# ESTABLISH BENCHMARK
# ======================
# use original 93 features 
set.seed(119)
intrain_feat_0 <- createDataPartition(trainset$target, p = 0.8, list = F)
trainset_feat_0 <- trainset[intrain_feat_0, -94]  # 39611 x 93
dim(trainset_feat_0)
label_trainset_feat_0 <- as.factor(as.vector(trainset[intrain_feat_0, 94]))  # 39611 x 1
length(label_trainset_feat_0)
testset_feat_0 <- trainset[-intrain_feat_0, ]


ctrl_rf_feat_0 <- trainControl(method = 'cv', 
                               number = 5, 
                               verboseIter = T,
                               savePredictions = T,
                               classProbs = T, 
                               summaryFunction = LogLossSummary)

grid_rf_feat_0 <- expand.grid(mtry = c(12, 15, 18, 20))
grid_rf_feat_0 <- expand.grid(mtry = 18)

ptm <- proc.time()
model_rf_feat_0 <- train(x = trainset_feat_0,
                         y = label_trainset_feat_0, 
                         method = 'rf', 
                         metric = 'LogLoss', 
                         maximize = F,
                         preProcess = c("scale", "center"),
                         tuneGrid = grid_rf_feat_0, 
                         trControl = ctrl_rf_feat_0, 
                         ntree = 500)
model_rf_feat_0$time <- proc.time() - ptm

varimp_rf_feat_0 <- varImp(model_rf_feat_0$finalModel)
varImpPlot(model_rf_feat_0$finalModel)




### RUN NO. 1
### BEST RESULTS: 0.5816251, mtry = 18, on average belw 0.6
### RESULTS 
# > model_rf_feat_0
# Random Forest 
# 
# 39611 samples
# 93 predictor
# 9 classes: 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9' 
# 
# No pre-processing
# Resampling: Cross-Validated (5 fold) 
# Summary of sample sizes: 31690, 31689, 31689, 31689, 31687 
# Resampling results across tuning parameters:
#   
#   mtry  LogLoss    LogLoss SD
# 12    0.5908903  0.01485675
# 15    0.5850496  0.01692714
# 18    0.5816251  0.01637539
# 20    0.5866679  0.01337185
# 
# LogLoss was used to select the optimal model using  the smallest value.
# The final value used for the model was mtry = 18. 


### RUN NO. 2
### BEST RESULT : 0.5918995, scale and center did not really help
### ADD PREPCOCESS (scale and center); USE ONLY mtry = 18
# > model_rf_feat_0
# Random Forest 
# 
# 39611 samples
# 93 predictor
# 9 classes: 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9' 
# 
# Pre-processing: scaled (93), centered (93) 
# Resampling: Cross-Validated (5 fold) 
# Summary of sample sizes: 31690, 31687, 31688, 31691, 31688 
# Resampling results
# 
# LogLoss    LogLoss SD
# 0.5918995  0.01407425
# 
# Tuning parameter 'mtry' was held constant at a value of 18



