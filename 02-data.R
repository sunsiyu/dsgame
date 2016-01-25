# ===================
# SET SEED
# ===================
set.seed(114)

# ===================
# DATA IMPORT
# ===================
trainfile <- dir(".", "train.csv", recursive = T)
testfile <- dir(".", "test.csv", recursive = T)
train <- read.table(trainfile, header=T, sep=",")
train <- train[, -1]  # remove id
cat(c("original dimension of train.csv is: ", dim(train), "..."), fill = T)

# ===================
# DATA SPLIT
# ===================
intrain <- createDataPartition(train$target, p = 0.8, list = FALSE)
trainset <- train[intrain, ]
testset <- train[-intrain, ]
cat(c("Splited training set dimension: ", dim(trainset), "..."), fill = T)

### class distribution checks
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
# Implement Loss Function and Summary here 
# note: the function only takes in matrices
LogLoss <- function(actual, predicted, eps = 1e-15) {
  predicted <- pmax(pmin(predicted, 1 - eps), eps)
  -sum(actual*log(predicted))/nrow(actual)
}

checkLogLoss <- function(model, data) {
  # create dummy predictions and compare with fitted model
  pred <- as.matrix(predict(model, newdata = data, type = 'prob'))
  dummy.fit <- dummyVars(~ target, data = data, levelsOnly = T)  
  truth <- predict(dummy.fit, newdata = data)  # predict ground truth using validation set
  LogLoss(truth, pred)
}

LogLossSummary <- function(data, lev = NULL, model = NULL) {
  
  LogLoss <- function(actual, pred, eps = 1e-15) {
    stopifnot(all(dim(actual) == dim(pred)))
    pred[pred < eps] <- eps
    pred[pred > 1 - eps] <- 1 - eps
    -sum(actual * log(pred)) / nrow(pred)
  }
  
  if (is.character(data$obs)) {
    data$obs <- factor(data$obs, levels = lev)
  }
  
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
