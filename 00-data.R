# ===================
# LOAD PACKAGES
# ===================
library(ggplot2)
library(dplyr)
library(mlbench)
library(DMwR)
library(tree)
library(rpart)
library(caret)
library(randomForest)
library(adabag)
library(gbm)
library(doParallel)
library(FSelector)
library(glmnet)
library(party)
library(kernlab)
library(xgboost)
# library(doMC) #not available for windows

# ===================
# SET SEED
# ===================
set.seed(114)
trainfile <- dir(".", "train.csv", recursive = T)
testfile <- dir(".", "test.csv", recursive = T)
samplesubmission <- dir(".", "sampleSubmission.csv", recursive = T)
samplesubmission <- read.table(samplesubmission, header = T, sep=",")
train <- read.table(trainfile, header=T, sep=",")
testing <- read.table(testfile, header=T, sep=",")
train <- train[, -1]  # remove id
testing <- testing[, -1] # remove id
cat(c("original dimension of train.csv is: ", dim(train), "..."), fill = T)

# ===================
# DATA SPLIT
# ===================
intrain <- createDataPartition(train$target, p = 0.8, list = FALSE)
trainset <- train[intrain, ]
testset <- train[-intrain, ]
cat(c("Splited training set dimension: ", dim(trainset), "..."), fill = T)
### some checks
# nrow(trainset) /nrow(train)
# prop.table(table(train$target))
# prop.table(table(trainset$target))

# ================================
# DATA SPLIT - FEATURE SELECTION
# ================================
# set.seed(119)
# intrain_feat <- createDataPartition(trainset$target, p = 0.8, list = FALSE)
# trainset_feat <- trainset[intrain_feat, ]
# testset_feat <- trainset[-intrain_feat, ]
# cat(c("Splited training set dimension: ", dim(trainset_feat), "..."), fill = T)
### some checks
# nrow(trainset) /nrow(train)
# prop.table(table(train$target))
# prop.table(table(trainset$target))


# ===================
# CLEAN UP
# ===================
rm(trainfile, testfile)

# ===================
# UTILITIES
# ===================
# implement log loss function and test it with test case from here: 
# note: the function only takes in matrices
LogLoss <- function(actual, predicted, eps = 1e-15) {
  predicted <- pmax(pmin(predicted, 1 - eps), eps)
  -sum(actual*log(predicted))/nrow(actual)
}

# actual <- matrix(data = c(0, 1, 0, 1, 0, 0, 0, 0, 1), nrow = 3)
# pred <- matrix(data = c(0.2, 0.7, 0.1, 0.6, 0.2, 0.2, 0.6, 0.1, 0.3), nrow = 3, byrow = T)
# LogLoss(actual, pred)  # should be 0.6904911, checks

# create function to compute logloss on validation set using ground truth
checkLogLoss <- function(model, data) {
  # LogLoss Function
  LogLoss <- function(actual, predicted, eps = 1e-15) {
    predicted <- pmax(pmin(predicted, 1 - eps), eps)
    -sum(actual*log(predicted))/nrow(actual)
  }
  # create dummy predictions and compare with fitted model
  pred <- as.matrix(predict(model, newdata = data, type = 'prob'))
  dummy.fit <- dummyVars(~ target, data = data, levelsOnly = T)  
  truth <- predict(dummy.fit, newdata = data)  # predict ground truth using validation set
  LogLoss(truth, pred)
}


# create custom logloss summary function for use with caret cross validation
LogLossSummary <- function(data, lev = NULL, model = NULL) {
  # this is slightly different from function above as above function leads to errors
  LogLoss <- function(actual, pred, eps = 1e-15) {
    stopifnot(all(dim(actual) == dim(pred)))
    pred[pred < eps] <- eps
    pred[pred > 1 - eps] <- 1 - eps
    -sum(actual * log(pred)) / nrow(pred)
  }
  if (is.character(data$obs)) data$obs <- factor(data$obs, levels = lev)
  pred <- data[, 'pred']
  obs <- data[, 'obs']
  is.na <- is.na(pred)
  pred <- pred[!is.na]
  obs <- obs[!is.na]
  data <- data[!is.na, ]
  class <- levels(obs)
  
  if (length(obs) + length(pred) == 0) {
    out <- rep(NA, 2)
  } else {
    probs <- data[, class]
    actual <- model.matrix(~ obs - 1)
    out <- LogLoss(actual = actual, pred = probs)
  }
  names(out) <- c('LogLoss')
  
  if (any(is.nan(out))) out[is.nan(out)] <- NA
  
  out
}
