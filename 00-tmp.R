# ===================
# LOAD PACKAGES
# ===================
library(dplyr, quietly = T)
library(ggplot2, quietly = T)
library(caret, quietly = T)
library(randomForest, quietly = T)
library(kernlab, quietly = T)
library(AppliedPredictiveModeling, quietly = T)

# ===================
# UTILITIES
# ===================
# function to impute
impute <- function(df, training) {
  # factors
  df[is.na(df$v9), "v9"] <- names(which.max(table(training$v9)))
  df[is.na(df$v20), "v20"] <- names(which.max(table(training$v20)))
  df[is.na(df$v41), "v41"] <- names(which.max(table(training$v41)))
  df[is.na(df$v31), "v31"] <- names(which.max(table(training$v31)))
  df[is.na(df$v36), "v36"] <- names(which.max(table(training$v36)))
  
  # numeric
  df[is.na(df$v17), "v17"] <- median(training$v17, na.rm=T)
  df[is.na(df$v39), "v39"] <- median(training$v39, na.rm=T)
  df[is.na(df$v18), "v18"] <- median(training$v18, na.rm=T)
  
  return(df)
}

# ===================
# LOAD DATA
# ===================
cclass <- c("factor", "numeric", "numeric", "factor", "factor", "factor", 
            "factor", "numeric", "factor", "factor", "numeric", "factor", 
            "factor", "factor", "numeric", "integer", "numeric", "factor", "factor")

training <- read.table("Training.csv", header = T, sep=";", colClasses = cclass, dec = ",")
testing <- read.table("Validation.csv", header = T, sep=";", colClasses = cclass, dec = ",")

training <- tbl_df(training)
testing <- tbl_df(testing)

# ==============================
# DATA CLEAN & FEATURE SELECTION
# ==============================

# Check zero covariates
nsv <- nearZeroVar(training, saveMetrics = TRUE)

# v7 is an identical column of labels, remove v7
p <- ggplot(training, aes(v7, fill=classLabel))
p + geom_histogram()
training <- select(training, -v7)
testing <- select(testing, -v7)

# Check feature with more than 50% missing value
sum(is.na(training$v35)) / nrow(training)
training <- select(training, -v35)
testing <- select(testing, -v35)


# Add Feature - nacount represent info. of NA
training <- mutate(training, nacount = is.na(v17) + is.na(v20) + is.na(v41) + 
                     is.na(v31) + is.na(v36) + is.na(v39) + is.na(v18),
                   fac_nacount = cut(nacount, c(0, 1, 2, 3, 8), right = F))
prop.table(table(training$fac_nacount))

testing <- mutate(testing, nacount = is.na(v17) + is.na(v20) + is.na(v41) + 
                    is.na(v31) + is.na(v36) + is.na(v39) + is.na(v18),
                  fac_nacount = cut(nacount, c(0, 1, 2, 3, 8), right = F))

# Imputation (missing value)
testing <- impute(testing, training)
training <- mutate_each(training, funs(replace(., which(is.na(.)), 
                                               ifelse(is.factor(.), names(which.max(table(.))), median(., na.rm=TRUE)))))

# Standardization
preObj <- preProcess(training, method = c("center", "scale"))
training <- predict(preObj, training)
testing <- predict(preObj, testing)



# ===================
# DATA SPLIT
# ===================
set.seed(1204)

# Remove non-features
training <- select(training, -nacount)
testing <- select(testing, -nacount)
training <- as.data.frame(training)
testing <- as.data.frame(testing)

## Splite training and testing sets (used for Tuning only)
# inTrain <- createDataPartition(y = training$classLabel, p = 0.7, list = F)
# toTrain <- training[inTrain, -17]
# v_label_train <- training[inTrain, 17]
# toTest <- training[-inTrain, -17]
# v_label_test <- training[-inTrain, 17]

toTrain <- training[, -17]
v_label_train <- training[, 17]
toTest <- testing[, -17]
v_label_test <- testing[, 17]


# ===================
# TRAINING
# ===================

## Decision Tree
modelTree <- train(classLabel ~ ., data=training, method = "rpart", tuneLength=60)
predictTree = predict(modelTree, testing)
confusionMatrix(predictTree,testing$classLabel)

## Stochastic Gradient Boosting
modelBoost <- train(classLabel ~ ., data = training, method="gbm", trControl = trainControl(method = "cv", number = 3),verbose=FALSE)
predictBoost <- predict(modelBoost,testing)
confusionMatrix(predictBoost,testing$classLabel)

## Random Forest
rf <- randomForest(x = toTrain, 
                   y = v_label_train, 
                   xtest = toTest, 
                   ytest = v_label_test, 
                   ntree = 500,
                   mtry = 5,
                   do.trace = 5,
                   keep.forest=TRUE)
rf
# ===================
# PREDICTION RESULTS
# ===================
prediction <- predict(rf, newdata = toTrain)


# ===================
# VISUALIZATION
# ===================

# Exploratory plot of random two features
p <- ggplot(training, aes(v35, v27, color = classLabel))
p + geom_jitter()

## Scatterplot Matrix
transparentTheme(trans = 0.4)
featurePlot(x = training[, 1:4],
            y = training$classLabel, 
            plot = "pairs", 
            ## Add a key at the top
            auto.key = list(columns = 3))
