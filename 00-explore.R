# ===================
# LOAD PACKAGES
# ===================
library(ggplot2)
library(dplyr)
library(caret)
library(randomForest)
library(DMwR)

# ===================
# SET SEED
# ===================
set.seed(114)

# ===================
# LOAD DATA
# ===================
trainfile <- dir(".", "train.csv", recursive = T)
testfiles <- dir(".", "test.csv", recursive = T)
samplesubmission <- dir(".", "sampleSubmission.csv", recursive = T)
samplesubmission <- read.table(samplesubmission, header = T, sep=",")
train <- read.table(trainfile, header=T, sep=",")
# in total 61878 x 95
# 1 id, 93 features, 1 label
# all features are integers
# range of features varies
# label with multiple classes
# proportion of each class not equal, 2,6,8,3,9,7,

train <- train[, -1]  # remove id
labels <- train[, 94]

# ===================
# LABEL EXPLORE
# ===================
# distribution of classes
nclasses <- table(train$target)
max(nclasses) / min(nclasses)
prop.table(table(train$target))
qplot(factor(target), data=train, geom="bar", fill = I("darkgray")) + coord_flip()
barplot(table(train$target), horiz = T)

# ===================
# DATA EXPLORE
# ===================
### features: total non-zero values in each feature
## TODO: look at histgram / distribution of non-zero values
npos <- sapply(train[, -94], function(x) sum(x>0)/nrow(train))
barplot(npos, horiz = T)
range(npos)
hist(npos, breaks = 50)

l_npos <- vector("list", 9)
for (i in 1:9) {
  tmp <- train[train$target == levels(train$target)[i], ]
  l_npos[[i]] <- sapply(tmp[, -94], function(x) sum(x>0)/nrow(tmp))
}
l_npos <- do.call(rbind, l_npos)
boxplot(l_npos)

### Check for duplicates(no duplicates)
which(duplicated(train[, -94]))

### check for missing values (no NAs)
which(is.na(train))


# ==========================================
# DATA RESAMPLING (imbalanced data needed)
# ==========================================
### downsample, upsample, sampling methods for imbalanced data
down_train <- downSample(train[, -94], train[, 94])
names(down_train)[94] <- "target"
up_train <- upSample(train[, -94], train[, 94])
names(up_train)[94] <- "target"
smote_train <- SMOTE(target ~ ., data = train)

l_train <- list(down_train, up_train, smote_train)



# ===================
# TRAINING
# ===================

for (i in 1:4){
  lfit <- lapply(l_train, function(x) randomForest(target ~ ., data=x, importance=TRUE, ntree=100))
}

# create a random forest model using the target field as the response and all 93 features as inputs
fit <- randomForest(Class ~ ., data=train2, importance=TRUE, ntree=100)

# create a dotchart of variable/feature importance as measured by a Random Forest
varImpPlot(fit)

# use the random forest model to create a prediction
pred <- predict(fit,test,type="prob")
submit <- data.frame(id = test$id, pred)

# data split
inTrain <- createDataPartition(train$target, )

# ===================
# PREDICTION RESULTS
# ===================
# use the random forest model to create a prediction
test <- read.table(testfiles, header=T, sep=",")
id <- test[, 1]
test <- test[,-1]

pred_prob <- predict(lfit[[2]], test, type="prob")
pred_class <- predict(lfit[[2]], test)

submit <- data.frame(id = id, pred_class)
submit$x <- 1
submit <- reshape(submit, idvar = "id", timevar = "pred_class", direction = "wide")
submit <- submit[, c(1, 10, 4, 6, 2, 8, 3, 9, 7, 5)]
submit[is.na(submit)] <- 0
names(submit) <- names(samplesubmission)
#or
submit <- data.frame(id = id, pred_prob)
write.csv(submit, file = "submit-02.csv", row.names = FALSE)
