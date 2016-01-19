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

# ===================
# LOAD DATA
# ===================
trainfile <- dir(".", "train.csv", recursive = T)
testfile <- dir(".", "test.csv", recursive = T)
samplesubmission <- dir(".", "sampleSubmission.csv", recursive = T)
samplesubmission <- read.table(samplesubmission, header = T, sep=",")
train <- read.table(trainfile, header=T, sep=",")
testing <- read.table(testfile, header=T, sep=",")
dim(train)  # 61878 x 95
dim(testing)  # 144368 x 94
# str(train)
# 1 id, 93 features, 1 label
# all features are integers
# range of features varies
# label with multiple classes
# proportion of each class not equal, 2,6,8,3,9,7,


train <- train[, -1]  # remove id
testing <- testing[, -1] # remove id
# ===================
# DATA SPLIT
# ===================
intrain <- createDataPartition(train$target, p = 0.8, list = FALSE)
trainset <- train[intrain, ]
testset <- train[-intrain, ]
### some checks
# nrow(trainset) /nrow(train)
# prop.table(table(train$target))
# prop.table(table(trainset$target))

# ===================
# LABEL EXPLORE
# ===================
# distribution of classes
nclasses <- table(train$target)
max(nclasses) / min(nclasses)  # 8.357698
proptbl <- prop.table(table(train$target))
proptbl <- as.data.frame(proptbl)
colnames(proptbl) <- c("class", "frequency")

# just plot total number
qplot(factor(target), data=train, geom="bar", fill = I("darkgray")) + coord_flip()
barplot(table(train$target), horiz = T)

# ggplot proptbl
p1 <- ggplot(proptbl, aes(reorder(class, frequency), frequency)) + 
  geom_bar(stat="identity", width = 0.1, fill = "skyblue") + 
  coord_flip() + 
  ggtitle("Class Imbalance")

# ===================
# DATA EXPLORE
# ===================

### Check for duplicates(no duplicates)
which(duplicated(train[, -94]))

### check for missing values (no NAs)
which(is.na(train))



### features: total non-zero values in each feature
## TODO: look at histgram / distribution of non-zero values
get_feat_zeroratio <- function(df, order = T) {
  stopifnot(is.data.frame(df))
  feat_zeroratio <- apply(df, 2, function(x) sum(x==0) / nrow(df))
  
  if (order) {
    feat_zeroratio <- feat_zeroratio[order(-feat_zeroratio)]
  }
  feat_zeroratio <- as.data.frame(feat_zeroratio)
  feat_zeroratio$feature <- rownames(feat_zeroratio)
  rownames(feat_zeroratio) <- 1:nrow(feat_zeroratio)
  return(feat_zeroratio)
}

train_zeroratio <- get_feat_zeroratio(train[, -94]) 
# ggplot feature zero ratio in train 
p2 <- ggplot(train_zeroratio, aes(reorder(feature, feat_zeroratio), feat_zeroratio)) + 
  geom_bar(stat="identity", width = 0.1, fill = "black") + 
  coord_flip() + 
  ggtitle("Feature Zero Ratios")

test_zeroratio <- get_feat_zeroratio(testing)
p3 <- ggplot(test_zeroratio, aes(reorder(feature, feat_zeroratio), feat_zeroratio)) + 
  geom_bar(stat="identity", width = 0.1, fill = "black") + 
  coord_flip() + 
  ggtitle("Feature Zero Ratios - toSubmit.csv")

# check similar zero ratios distribution in features between train set and to submit set
length(unique(train_zeroratio[1:20, "feature"], test_zeroratio[1:20, "feature"]))


# Relation between zeroratio and class
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


# original feature rank with train
correlationMatrix <- cor(train[, -94])
high_corr <- findCorrelation(correlationMatrix, cutoff = 0.8) # feat_45
nzv <- nzv(train[, -94], saveMetrics = T)
names_nzv <- rownames(nzv[nzv$nzv, ])
names_zeroratio <- train_zeroratio[train_zeroratio$feat_zeroratio > 0.9, "feature"]
names_out <- unique(names_nzv, names_zeroratio)
length(names_out)


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
