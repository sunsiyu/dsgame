# ===================
# LOAD PACKAGES
# ===================
library(ggplot2)
library(dplyr)
library(mlbench)
library(caret)
library(randomForest)
library(DMwR)
library(FSelector)

# ===================
# SET SEED
# ===================
set.seed(5)

# ======================
# FEATURE SELECTION
# ======================
data("PimaIndiansDiabetes")  # 768 x 9, label:col9

correlationMatrix <- cor(PimaIndiansDiabetes[, 1:8])
correlationMatrix

high_corr <- findCorrelation(correlationMatrix, cutoff = 0.7)
high_corr  # integer(0)

control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
model <- train(diabetes ~ ., 
               data = PimaIndiansDiabetes, 
               # LVQ
               method = "lvq", 
               preProcess = "scale", 
               trControl = control)
importance <- varImp(model, scale = F)
importance
plot(importance)

control <- rfeControl(functions = rfFuncs, method = "cv", number = 10)
# Backwards Feature Selection
results <- rfe(PimaIndiansDiabetes[, 1:8], 
               PimaIndiansDiabetes[, 9], 
               sizes = c(1:8), 
               rfeControl = control)

results
predictors(results)  # return the feature names
plot(results, type = c("g", "o"))

# ============================
# FEATURE RANK with FSelector
# ============================
chi.squared(diabetes ~ ., data = PimaIndiansDiabetes)
linear.correlation(diabetes ~ ., data = PimaIndiansDiabetes) # error: must be numeric

rank.correlation(diabetes ~ ., data = PimaIndiansDiabetes) # error: must be numeric
information.gain(diabetes ~ ., data = PimaIndiansDiabetes)
gain.ratio(diabetes ~ ., data = PimaIndiansDiabetes)
symmetrical.uncertainty(diabetes ~ ., data = PimaIndiansDiabetes)

oneR(diabetes ~ ., data = PimaIndiansDiabetes)
#  1=mean decrease in accuracy, 2=mean decrease in node impurity
random.forest.importance(diabetes ~ ., data = PimaIndiansDiabetes, importance.type = 2)


# =======================================
# FEATURE SUBSET SELECTION with FSelector
# =======================================

cfs(diabetes ~ ., data = PimaIndiansDiabetes)
consistency(diabetes ~ ., data = PimaIndiansDiabetes)

# best.first.search(attributes, eval.fun)
# exhaustive.search(attributes, eval.fun)
# forward.search(attributes, eval.fun)
# backward.search(attributes, eval.fun)
# hill.climbing.search(attributes, eval.fun)



