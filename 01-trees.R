# ===================
# LOAD PACKAGES
# ===================
library(dplyr, quietly = T)
library(ggplot2, quietly = T)
library(caret, quietly = T)
library(randomForest, quietly = T)
library(kernlab, quietly = T)
library(AppliedPredictiveModeling, quietly = T)
library(tree)
library(rpart)
library(adabag)
library(gbm)

# ===================
# SET SEED
# ===================
set.seed(114)

# ===================
# LOAD DATA
# ===================
trainfile <- dir(".", "train.csv", recursive = T)
testfile <- dir(".", "test.csv", recursive = T)
samplefile <- dir(".", "sampleSubmission.csv", recursive = T)

train <- read.table(trainfile, header=T, sep=",")
# in total 61878 x 95
# 1 id, 93 features, 1 label
# all features are integers
# proportion of each class not equal, 2,6,8,3,9,7
# an imbalanced multi-classification problem

test <- read.table(testfile, header=T, sep=",")
# in total 144368 x 94, more than training dataset
# 1st id col


# ===================
# DATA EXPLORE
# ===================
# TODO: explorative statistic summaries
# range of each feature
# mean, variance, per(non-zero values)
summary(train)
summary(test)


# ===================
# DATA SPLIT
# ===================
strain <- train[sample(nrow(train), 6000, replace=FALSE),]
stest <- train[sample(nrow(train), 2000, replace=FALSE),]

# ===================
# TRAINING
# ===================
# just fit without any modification on the features to get a feel
# fit_rf: random forest
# fit_tree: a simple decision tree
# fit_rpart: another simple decision tree

# Use the tuneRF function to determine an ideal value for the mtry parameter
mtry <- tuneRF(strain[,1:93], strain[,94], mtryStart=1, ntreeTry=50, stepFactor=2, improve=0.05,
               trace=TRUE, plot=TRUE, doBest=FALSE)
fit_rf    <- randomForest(target ~ ., 
                          data = train[, -1], 
                          importance=TRUE, 
                          ntree=100)
fit_tree  <- tree(as.factor(target) ~ ., data=strain)
fit_rpart <- rpart(as.factor(target) ~ ., data=strain, method="class")
fit_gbm   <- gbm(target ~ ., 
                 data = strain, 
                 distribution = "multinomial", 
                 n.trees=1000, 
                 shrinkage=0.05, 
                 interaction.depth=12, 
                 cv.folds=2)

trees <- gbm.perf(fit5)
fit5.stest <- predict(fit5, stest, n.trees=trees, type="response")
fit5.stest <- as.data.frame(fit5.stest)
names(fit5.stest) <- c("Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9")
fit5.stest.pred <- rep(NA,2000)
for (i in 1:nrow(stest)) {
  fit5.stest.pred[i] <- colnames(fit5.stest)[(which.max(fit5.stest[i,]))]}

# ===================
# EVALUATION
# ===================
# create a dotchart of variable/feature importance as measured by a Random Forest
varImpPlot(fit)


# Create a confusion matrix of predictions vs actuals
table(fit1.pred,stest$target)
confusionMatrix(pred, stest$class)

# Determine the error rate for the model
fit1$error <- 1-(sum(fit1.pred==stest$target)/length(stest$target))
fit1$error

# ===================
# VISUALIZE
# ===================
# Plot the decision tree
par(xpd=TRUE)
plot(fit2, compress=TRUE)
title(main="rpart")
text(fit2)

# ===================
# PREDICTION
# ===================

pred <- predict(fit, stest, type = "prob")
pred <- predict(fit_tree, stest, type = "class")
submit <- data.frame(id = test$id, pred)
write.csv(submit, file = "submit-x.csv", row.names = FALSE)


# ===================
# COMPUTATION TIME
# ===================
# Begin recording the time it takes to create the model
ptm <- proc.time()
# Create a bagging model using the target field as the response and all 93 features as inputs (.)
fit_bag <- bagging(target ~ ., data=strain, mfinal=50)
# Finish timing the model
fit_bag$time <- proc.time() - ptm
# Test the baggind model on the holdout test dataset
pred_bag <- predict(fit_bag, stest, newmfinal = 50)


# ===================
# CONCLUSION
# ===================

# Decision trees do not work very well, too many features, recursive binary 
# partitioning is over-simplifying the model and miss some classes completely.
