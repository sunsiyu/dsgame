# other feature engineering

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
