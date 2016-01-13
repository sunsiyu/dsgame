##https://github.com/eugeneyan/kaggle_otto.git


############################################################################
### load libraries
############################################################################
library(caret)
library(doParallel)
detectCores()
registerDoParallel(detectCores() - 1) 
getDoParWorkers()

############################################################################
### load data sets and create train/validation split
############################################################################
# load training data
train <- read.csv("train.csv", header = T)
train$id <- NULL  # remove ID
set.seed(668)
train <- train[sample(nrow(train)), ]  # shuffle data

# load testing data
test <- read.csv("test.csv", header = T)
test.id <- test$id
test$id <- NULL  # remove ID

# partition into training and validation set
set.seed(668)
in.train <- createDataPartition(y = train$target, p = 0.80, list = F)  # use this to train model
in.train <- in.train[1:49506]  # this makes it a vector

############################################################################
### useful functions for checking logloss and creating submissions
############################################################################
# implement log loss function and test it with test case from here: 
# http://www.kaggle.com/c/emc-data-science/forums/t/2149/is-anyone-noticing-difference-betwen-validation-and-leaderboard-error
# note: the function only takes in matrices
LogLoss <- function(actual, predicted, eps = 1e-15) {
  predicted <- pmax(pmin(predicted, 1 - eps), eps)
  -sum(actual*log(predicted))/nrow(actual)
}

actual <- matrix(data = c(0, 1, 0, 1, 0, 0, 0, 0, 1), nrow = 3)
pred <- matrix(data = c(0.2, 0.7, 0.1, 0.6, 0.2, 0.2, 0.6, 0.1, 0.3), 
               nrow = 3, byrow = T)

LogLoss(actual, pred)  # this should be 0.6904911 if the function is working correction


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


# create function to create submissions
# note: file should be a string
submit <- function(model, data, file) {
  
  # create predictions and write out to csv.file
  pred <- predict(model, newdata = data, type = 'prob', na.action = NULL)
  submission <- data.frame(id = test.id, pred)
  write.csv(submission, file = file, row.names = F)
}

############################################################################
### random forest
############################################################################
# set params for fit control
ctrl <- trainControl(method = 'cv', number = 5, verboseIter = T, classProbs = T, 
                     summaryFunction = LogLossSummary)

rf.grid <- expand.grid(mtry = c(6, 9, 12))

rf.fit <- train(target ~., data = train[in.train, ], method = 'rf', 
                metric = 'LogLoss', maximize = F,
                tuneGrid = rf.grid, trControl = ctrl, ntree = 500)

# create predictions
rf.pred <- as.matrix(predict(rf.fit, newdata = train[-in.train, ], type = 'prob'))

# compute log.loss  
checkLogLoss(rf.fit, train[-in.train, ])  # logloss = 0.56234

############################################################################
### gbm
############################################################################
# set params for fit control
ctrl <- trainControl(method = 'cv', number = 5, verboseIter = T, classProbs = T, 
                     summaryFunction = LogLossSummary)

gbm.grid <- expand.grid(interaction.depth = 10,
                        n.trees = (2:100) * 50,
                        shrinkage = 0.005)

gbm.fit <- train(target ~., data = train[in.train, ], 
                 method = 'gbm', distribution = 'multinomial', 
                 metric = 'LogLoss', maximize = F, 
                 tuneGrid = gbm.grid, trControl = ctrl,
                 n.minobsinnode = 4, bag.fraction = 0.9)

checkLogLoss(gbm.fit, train[-in.train, ])  # log.loss = 0.509993


# good practice to close connections when done
showConnections()
closeAllConnections()
showConnections()

#-------------------------------------------------------------------------------



############################################################################
### load libraries
############################################################################
library(caret)
library(doParallel)
detectCores()
registerDoParallel(detectCores() - 1) 
getDoParWorkers()

# packages for models
library(randomForest)
library(glmnet)
library(party)

# packages for data munging and visualization
library(dplyr)
library(ggplot2)

############################################################################
### feature selection with glmnet (did not help much)
############################################################################
# glmnet
# note: caret is unable to let glmnet choose lambda range 
x <- model.matrix(target ~., data = train[in.train, ])[, -1]
y <- train[in.train, ]$target

# find best lambda at alpha = 0.5
glmnet.cv <- cv.glmnet(x = x, y = y, 
                       alpha = 0.5, family = 'multinomial', 
                       nfold = 5, parallel = T)

glmnet.cv$lambda.min  # 0.0002909891
glmnet.cv$lambda.1se  # 0.0008886387


# set params for fit control
ctrl <- trainControl(method = 'cv', number = 5, verboseIter = T, classProbs = T, 
                     summaryFunction = LogLossSummary)

# use lambda found from above in glmnet.grid to use caret.glmnet
glmnet.grid <- expand.grid(alpha = (1:5) * 0.2, 
                           lambda = (1:5) * 0.002)

glmnet.fit <- train(target ~., data = train[in.train, ], method = 'glmnet', 
                    metric = 'LogLoss', maximize = F, tuneGrid = glmnet.grid, trControl = ctrl)

# LogLoss was used to select the optimal model using  the smallest value.
# The final values used for the model were alpha = 0.2 and lambda = 0.002. 

# examine coefficients
coef(glmnet.fit$finalModel, glmnet.fit$bestTune$.lambda)
coef(glmnet.fit$finalModel, 0.002)  # doesn't seem like we can exclude any features


# find best lambda at alpha = 1 
glmnet.grid <- expand.grid(alpha = 1, 
                           lambda = (1:5) * 0.002)

glmnet.fit <- train(target ~., data = train[in.train, ], method = 'glmnet', 
                    metric = 'LogLoss', maximize = F, tuneGrid = glmnet.grid, trControl = ctrl)

coef(glmnet.fit$finalModel, 0.002)  # looks like a dead end

############################################################################
### rf and gbm to examine top features (cforest was unusable)
############################################################################
# small rf
rf.fit <- randomForest(target ~., data = train[in.train, ], 
                       mtry = 9, nodesize = 5, ntree = 500, 
                       keep.forest = T, importance = T)

# rf variable importance
rf.imp <- importance(rf.fit, scale = T)
varImpPlot(rf.fit, n.var = 20)  # top 20 most impt variables

# extract 20 most important rf variables
rf.imp <- as.data.frame(importance(rf.fit, type = 1))
rf.imp$Vars <- row.names(rf.imp)
rf.20 <- rf.imp[order(-rf.imp$MeanDecreaseAccuracy),][1:20,]$Vars


# small gbm
gbm.grid <- expand.grid(interaction.depth = 10,
                        n.trees = 100,
                        shrinkage = 0.01)

gbm.fit <- train(target ~., data = train[in.train, ], 
                 method = 'gbm', distribution = 'multinomial', 
                 metric = 'LogLoss', maximize = F, 
                 tuneGrid = gbm.grid, trControl = ctrl,
                 n.minobsinnode = 4, bag.fraction = 0.9)

# gbm variable importance
varImp(gbm.fit, scale = T)

# extract 20 most important GBM variables
gbm.imp <- data.frame(varImp(gbm.fit)$importance)
gbm.imp$Vars <- row.names(gbm.imp)
gbm.20 <- gbm.imp[order(-gbm.imp$Overall),][1:20,]$Vars

# combine top features identified by rf and gbm
top.feats <- unique(c(gbm.20, rf.20))


# small cforest (unusable as it takes very long to calculate conditional importance)
cf.fit <- cforest(target ~., data = train[in.train, ], 
                  control = cforest_unbiased(mtry = 9, ntree = 200))

# see importance of CF variables (could not run)
cf.imp <- varimp(cf.fit, conditional = T, threshold = 0.8)  # runs fine when conditional = F

############################################################################
### creating new features with a small sample set
############################################################################
# take small sample for testing
set.seed(100)
feat <- train %>%
  sample_frac(0.1)

### create sum of rows, variance of rows, and number of columns filled
addAggFeatures <- function(data) {
  
  # add new features
  mutate(data, feat_sum = as.integer(rowSums(data[, 1:93])),  # count sum of features by row
         feat_var = as.integer(apply(data[, 1:93], 1, var)),  # variance of features by row
         feat_filled = as.integer(rowSums(data[, 1:93] != 0))  # count no. of non-empty features
  )
}

# add new features to feat with function
feat <- addAggFeatures(train)

# plot new variables
ggplot(data = feat, aes(x = target, y = feat_sum, col = target)) + 
  geom_boxplot()
ggplot(data = feat, aes(x = target, y = feat_var, col = target)) + 
  geom_boxplot() +
  scale_y_continuous(limits = c(0, 12))
ggplot(data = feat, aes(x = target, y = feat_filled, col = target)) + 
  geom_boxplot()


### for each row, "normalize" features by dividing by sum (not useful in LB score)
for (i in 1:93) {
  eval(parse(text = paste0('feat$feat_n_', i, ' <- feat[, i] / feat$feat_sum')))
}


### create +, -, *, / features using top 20 features (useful with original features but not useful after features were mean-standardized)
# select top 20 features
feat.20 <- train %>%
  select(c(feat_11, feat_60, feat_34, feat_90, feat_14,
           feat_15, feat_26, feat_40, feat_86, feat_75,
           feat_36, feat_42, feat_39, feat_69, feat_68,
           feat_67, feat_62, feat_25, feat_9, feat_24, 
           target))

# create +, -, *, / features
for (i in 1:19) {
  for (j in (i + 1) : 20) {
    var.x <- colnames(feat.20)[i]
    var.y <- colnames(feat.20)[j]
    var.new <- paste0(var.x, '_x_', var.y)
    eval(parse(text = paste0('feat.20$', var.new, ' <- feat.20[, i] * feat.20[, j]')))
  }
}

for (i in 1:19) {
  for (j in (i + 1) : 20) {
    var.x <- colnames(feat.20)[i]
    var.y <- colnames(feat.20)[j]
    var.new <- paste0(var.x, 'div', var.y)
    eval(parse(text = paste0('feat.20$', var.new, ' <- feat.20[, i] / feat.20[, j]')))
  }
}

for (i in 1:19) {
  for (j in (i + 1) : 20) {
    var.x <- colnames(feat.20)[i]
    var.y <- colnames(feat.20)[j]
    var.new <- paste0(var.x, 'plus', var.y)
    eval(parse(text = paste0('feat.20$', var.new, ' <- feat.20[, i] + feat.20[, j]')))
  }
}

for (i in 1:19) {
  for (j in (i + 1) : 20) {
    var.x <- colnames(feat.20)[i]
    var.y <- colnames(feat.20)[j]
    var.new <- paste0(var.x, 'min', var.y)
    eval(parse(text = paste0('feat.20$', var.new, ' <- feat.20[, i] - feat.20[, j]')))
  }
} 

### create factor for 1 when feat > 0 and 0 when feat = 0 (works poorly--do not use)
for (i in 1:93) {
  eval(parse(text = paste0('feat$feat_flag_', i, ' <- ifelse(feat[, i] == 0, 0, 1)')))
}

### features for difference from mean  (works well)
for (i in 1:93) {
  eval(parse(text = paste0('feat$feat_mean_', i, ' <- feat[, i] - mean(feat[, i])')))
}

feat <- feat[-c(1:93)]
str(feat)

### features after normalization  (works, but less well relative to difference from mean)
for (i in 1:93) {
  eval(parse(text = paste0('feat$feat_norm_', i, ' <- (feat[, i] - mean(feat[, i]))/sd(feat[, 1])')))
}

### remove original features
feat <- feat[-c(1:93)]
str(feat)

############################################################################
### add features to entire train data
############################################################################
### add agg features
train <- addAggFeatures(train)

### select top 20 features to add op features
feat.20 <- train %>%
  select(c(feat_11, feat_60, feat_34, feat_90, feat_14,
           feat_15, feat_26, feat_40, feat_86, feat_75,
           feat_36, feat_42, feat_39, feat_69, feat_68,
           feat_67, feat_62, feat_25, feat_9, feat_24, 
           target))

for (i in 1:19) {
  for (j in (i + 1) : 20) {
    var.x <- colnames(feat.20)[i]
    var.y <- colnames(feat.20)[j]
    var.new <- paste0(var.x, '_x_', var.y)
    eval(parse(text = paste0('feat.20$', var.new, ' <- feat.20[, i] * feat.20[, j]')))
  }
}  # multiplication

for (i in 1:19) {
  for (j in (i + 1) : 20) {
    var.x <- colnames(feat.20)[i]
    var.y <- colnames(feat.20)[j]
    var.new <- paste0(var.x, 'div', var.y)
    eval(parse(text = paste0('feat.20$', var.new, ' <- feat.20[, i] / feat.20[, j]')))
  }
}  # division

for (i in 1:19) {
  for (j in (i + 1) : 20) {
    var.x <- colnames(feat.20)[i]
    var.y <- colnames(feat.20)[j]
    var.new <- paste0(var.x, 'plus', var.y)
    eval(parse(text = paste0('feat.20$', var.new, ' <- feat.20[, i] + feat.20[, j]')))
  }
}  # addition

for (i in 1:19) {
  for (j in (i + 1) : 20) {
    var.x <- colnames(feat.20)[i]
    var.y <- colnames(feat.20)[j]
    var.new <- paste0(var.x, 'min', var.y)
    eval(parse(text = paste0('feat.20$', var.new, ' <- feat.20[, i] - feat.20[, j]')))
  }
}  # subtraction

# keep only created features
feat.20 <- (feat.20[-c(1:21)])


### extra clean up for division variables (due to errors from dividing 0 or dividing by 0)
str(feat.20[, 190: 220])
sapply(feat.20, function(x) sum(is.na(x)))

# cleaning up the Nan values from dividing 0
is.nan.data.frame <- function(x)
  do.call(cbind, lapply(x, is.nan))

feat.20[is.nan(feat.20)] <- 0

# cleaning up the Inf values from dividing by 0
feat.20[mapply(is.infinite, feat.20)] <- 0

# check the division variables
str(feat.20[, 190: 220])

### features for difference from mean
for (i in 1:93) {
  eval(parse(text = paste0('train$feat_mean_', i, ' <- train[, i] - mean(train[, i])')))
}

train <- train[-c(1:93)]

### features after normalization
for (i in 1:93) {
  eval(parse(text = paste0('train$feat_norm_', i, ' <- (train[, i] - mean(train[, i]))/sd(train[, i])')))
}

### number of filled features
train$feat_filled <- as.integer(rowSums(train[, 1:93] != 0))  # number of filled variables
train$feat_filled <- (train$feat_filled - mean(train$feat_filled))/sd(train$feat_filled)  # normalize

############################################################################
### add features to entire test data
############################################################################
### add agg features
test <- addAggFeatures(test)

### select top 20 features to add op features
feat.20 <- test %>%
  select(c(feat_11, feat_60, feat_34, feat_90, feat_14,
           feat_15, feat_26, feat_40, feat_86, feat_75,
           feat_36, feat_42, feat_39, feat_69, feat_68,
           feat_67, feat_62, feat_25, feat_9, feat_24, 
           target))

for (i in 1:19) {
  for (j in (i + 1) : 20) {
    var.x <- colnames(feat.20)[i]
    var.y <- colnames(feat.20)[j]
    var.new <- paste0(var.x, '_x_', var.y)
    eval(parse(text = paste0('feat.20$', var.new, ' <- feat.20[, i] * feat.20[, j]')))
  }
}  # multiplication

for (i in 1:19) {
  for (j in (i + 1) : 20) {
    var.x <- colnames(feat.20)[i]
    var.y <- colnames(feat.20)[j]
    var.new <- paste0(var.x, 'div', var.y)
    eval(parse(text = paste0('feat.20$', var.new, ' <- feat.20[, i] / feat.20[, j]')))
  }
}  # division

for (i in 1:19) {
  for (j in (i + 1) : 20) {
    var.x <- colnames(feat.20)[i]
    var.y <- colnames(feat.20)[j]
    var.new <- paste0(var.x, 'plus', var.y)
    eval(parse(text = paste0('feat.20$', var.new, ' <- feat.20[, i] + feat.20[, j]')))
  }
}  # addition

for (i in 1:19) {
  for (j in (i + 1) : 20) {
    var.x <- colnames(feat.20)[i]
    var.y <- colnames(feat.20)[j]
    var.new <- paste0(var.x, 'min', var.y)
    eval(parse(text = paste0('feat.20$', var.new, ' <- feat.20[, i] - feat.20[, j]')))
  }
}  # subtraction

# keep only created features
feat.20 <- (feat.20[-c(1:21)])


### extra clean up for division variables (due to errors from dividing 0 or dividing by 0)
str(feat.20[, 190: 220])
sapply(feat.20, function(x) sum(is.na(x)))

# cleaning up the Nan values from dividing 0
is.nan.data.frame <- function(x)
  do.call(cbind, lapply(x, is.nan))

feat.20[is.nan(feat.20)] <- 0

# cleaning up the Inf values from dividing by 0
feat.20[mapply(is.infinite, feat.20)] <- 0

# check the division variables
str(feat.20[, 190: 220])

### features for difference from mean
for (i in 1:93) {
  eval(parse(text = paste0('test$feat_mean_', i, ' <- test[, i] - mean(test[, i])')))
}

test <- test[-c(1:93)]

### features after normalization
for (i in 1:93) {
  eval(parse(text = paste0('test$feat_norm_', i, ' <- (test[, i] - mean(test[, i]))/sd(test[, i])')))
}

### number of filled features
test$feat_filled <- as.integer(rowSums(test[, 1:93] != 0))  # number of filled variables
test$feat_filled <- (test$feat_filled - mean(test$feat_filled))/sd(test$feat_filled)  # normalize



###----------------------------------------------------------------------------
############################################################################
### load libraries
############################################################################
library(xgboost)
library(methods)
library(caret)

############################################################################
### load train data and create matrices for xgb
############################################################################
train <- read.csv("train.csv", header = T)
train$id <- NULL  # remove ID
set.seed(668)
train <- train[sample(nrow(train)), ]  # shuffle data

# create target vector
train.y <- train$target
train.y <- gsub('Class_','', train.y)
train.y <- as.integer(train.y) - 1  #xgboost take features in [0, number of classes)

# create matrix of original features for train.x
train.x <- train
train.x$target <- NULL
train.x <- as.matrix(train.x)
train.x <- matrix(data = as.numeric(train.x), nrow = nrow(train.x), ncol = ncol(train.x))


############################################################################
### create useful functions (check log loss specific to xgb; requires matrix creation)
############################################################################
### create function to compute logloss on test set in one step
checkLogLoss2 <- function(model, xgbdata, traindata) {
  
  # LogLoss Function
  LogLoss <- function(actual, predicted, eps = 1e-15) {
    predicted <- pmax(pmin(predicted, 1 - eps), eps)
    -sum(actual*log(predicted))/nrow(actual)
  }
  
  # create predictions and dummy predictions and compare with fitted model
  pred <- predict(xgb.fit, newdata = xgbdata)
  pred <- t(matrix(pred, nrow = 9, ncol = length(pred)/9))  # prediction based on fitted model
  dummy.fit <- dummyVars(~ target, data = traindata, levelsOnly = T)
  truth <- predict(dummy.fit, newdata = traindata)  # ground truth
  LogLoss(truth, pred)
}


############################################################################
### try creating a small xgb model
############################################################################
# Set necessary parameter
xg.param <- list("objective" = "multi:softprob",
                 'eval_metric' = "mlogloss",
                 'num_class' = 9,
                 'eta' = 0.1,
                 'gamma' = 1,
                 'max.depth' = 10,
                 'min_child_weight' = 4,
                 'subsample' = 0.9,
                 'colsample_bytree' = 0.8,
                 'nthread' = 3)

# run cross validation
xgb.fit.cv <- xgb.cv(param = xg.param, data = train.x[in.train, ], label = train.y[in.train], 
                     nfold = 5, nrounds = 250)

# check best iteration
which(xgb.fit.cv$test.mlogloss.mean == min(xgb.fit.cv$test.mlogloss.mean))

# fit model on training set
xgb.fit <- xgboost(param = xg.param, data = train.x[in.train, ], 
                   label = train.y[in.train], nrounds = 250)

# check log loss on validation set
checkLogLoss2(xgb.fit, train.x[-in.train, ], train[-in.train, ])  # log.loss = 0.4627816

# fit model on full training data
xgb.fit <- xgboost(param = xg.param, data = train.x, 
                   label = train.y, nrounds = 250)


############################################################################
### check feature importance to create interaction features
# note: while this worked with the original features, it seemed to increase
# error rate when using scaled features. Thus, it was not used eventually.
############################################################################
# fit model on training set
xgb.fit <- xgboost(param = xg.param, data = train.x[in.train, ], 
                   label = train.y[in.train], nrounds = 100)

# check feature importance
xgb.importance(feature_names = names(train), model = xgb.fit)


############################################################################
### xgb using original + aggregated features
# aggregated features being row sum, row var, and no. of cols filled
############################################################################
# Set necessary parameter
xg.param <- list("objective" = "multi:softprob",
                 'eval_metric' = "mlogloss",
                 'num_class' = 9,
                 'eta' = 0.005,
                 'gamma' = 1,
                 'max.depth' = 10,
                 'min_child_weight' = 4,
                 'subsample' = 0.9,
                 'colsample_bytree' = 0.8,
                 'nthread' = 3)

# run cross validation
xgb.fit.cv <- xgb.cv(param = xg.param, data = train.x[in.train, ], label = train.y[in.train], 
                     nfold = 5, nrounds = 10000)

# check best iteration
cv.min <- min(xgb.fit.cv$test.mlogloss.mean)
cv.min.rounds <- which(xgb.fit.cv$test.mlogloss.mean == min(xgb.fit.cv$test.mlogloss.mean))  
# min = 0.469457 at nrounds 7483 7484

cv.rounds <- round(mean(which(xgb.fit.cv$test.mlogloss.mean == min(xgb.fit.cv$test.mlogloss.mean))))


# fit model on training set
xgb.fit <- xgboost(param = xg.param, data = train.x[in.train, ], 
                   label = train.y[in.train], nrounds = cv.rounds)

# check log loss on validation set
checkLogLoss2(xgb.fit, train.x[-in.train, ], train[-in.train, ])  
# log.loss = 0.4517306 (improvement of 0.0006185)


# fit model on full training data
xgb.fit <- xgboost(param = xg.param, data = train.x, 
                   label = train.y, nrounds = cv.rounds)
# LB score = 0.44085 

############################################################################
### xgb using original features (after mean-standardization) 
# gamma = 0.5, min_child = 4
############################################################################
# Set necessary parameter
xg.param <- list("objective" = "multi:softprob",
                 'eval_metric' = "mlogloss",
                 'num_class' = 9,
                 'eta' = 0.005,
                 'gamma' = 0.5,
                 'max.depth' = 10,
                 'min_child_weight' = 4,
                 'subsample' = 0.9,
                 'colsample_bytree' = 0.8,
                 'nthread' = 3)

xgb.fit.cv <- xgb.cv(param = xg.param, data = train.x, label = train.y, 
                     nfold = 5, nrounds = 10000)

# check best iteration
cv.min <- min(xgb.fit.cv$test.mlogloss.mean)
cv.min.rounds <- which(xgb.fit.cv$test.mlogloss.mean == min(xgb.fit.cv$test.mlogloss.mean))  
# min = 0.448826 at nrounds 7563

plot(xgb.fit.cv$test.mlogloss.mean[7000:8000])
cv.rounds <- round(mean(which(xgb.fit.cv$test.mlogloss.mean == min(xgb.fit.cv$test.mlogloss.mean))))  

# fit model on training set
xgb.fit <- xgboost(param = xg.param, data = train.x[in.train, ], 
                   label = train.y[in.train], nrounds = cv.rounds)

# check log loss on validation set
checkLogLoss2(xgb.fit, train.x[-in.train, ], train[-in.train, ])  
# log.loss = 0.4489771 

# fit model on full training data
xgb.fit <- xgboost(param = xg.param, data = train.x, 
                   label = train.y, nrounds = cv.rounds)
# LB score = 0.43609


############################################################################
### create predictions on test data
############################################################################
test <- read.csv("test.csv", header = T)
test.id <- test$id
test$id <- NULL

# create matrix of original features for test.x
test <- as.matrix(test)
test <- matrix(data = as.numeric(test), nrow = nrow(test), ncol = ncol(test))

# create predictions on test data 
# using original + aggregated features
xgb.pred <- predict(xgb.fit, test)
xgb.pred <- t(matrix(xgb.pred, nrow = 9, ncol = length(xgb.pred)/9))
xgb.pred <- data.frame(1:nrow(xgb.pred), xgb.pred)
names(xgb.pred) <- c('id', paste0('Class_',1:9))
write.csv(xgb.pred, file='xgb.pred.csv', quote=FALSE, row.names=FALSE)


# create predictions on test data 
# using original (after mean-standardization) features
test <- read.csv("test.csv", header = T)
test.id <- test$id
test$id <- NULL

# features for difference from mean
for (i in 1:93) {
  eval(parse(text = paste0('test$feat_mean_', i, ' <- test[, i] - mean(test[, i])')))
}
test <- test[-c(1:93)]

# create matrix of original features for test.x
test <- as.matrix(test)
test <- matrix(data = as.numeric(test), nrow = nrow(test), ncol = ncol(test))

# create predictions on test data 
# using original + aggregated features
xgb.pred <- predict(xgb.fit, test)
xgb.pred <- t(matrix(xgb.pred, nrow = 9, ncol = length(xgb.pred)/9))
xgb.pred <- data.frame(1:nrow(xgb.pred), xgb.pred)
names(xgb.pred) <- c('id', paste0('Class_',1:9))
write.csv(xgb.pred, file='xgb.pred2.csv', quote=FALSE, row.names=FALSE)















#----------