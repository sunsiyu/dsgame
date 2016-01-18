# ===================
# TRAIN PREPARATION
# ===================
labels <- as.factor(as.vector(trainset[, ncol(trainset)]))


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


control1 <- trainControl(method = "repeatedcv", 
                         number = 10,
                         repeats = 3)

trainset1 <- createDataPartition(trainset$Class, p = 0.1, list=F)
trainset1 <- trainset[trainset1, ]
labels1 <- trainset1$Class
# for whatever reason, outputs NaN for multi-class
model_nb <- train(Class ~ ., data = trainset1,
                  method = "nb", 
                  trControl = control1)

model_elm <- train(x = trainset[, -ncol(trainset)],
                  y = labels,
                  method = "elm", 
                  trControl = control1)

model_rf <- train(x = trainset[, -ncol(trainset)],
                  y = labels,
                  method = "rf", 
                  trControl = trainControl("oob"))



###############svm

set.seed(100)
trainset_59_feat_small <- createDataPartition(trainset_59_feat$target, p = 0.5, list=F)
trainset_59_feat_small <- trainset_59_feat[trainset_59_feat_small, ]

ctrl_svm <- trainControl(method = "cv", 
                         number = 5,
                         classProbs = T, 
                         savePredictions = T, 
                         verboseIter = T,
                         summaryFunction = LogLossSummary)
ptm4 <- proc.time()
model_svm1 = train(target ~ ., data = trainset_59_feat, 
                   method = "svmRadial", 
                   preProcess = c("center", "scale"),
                   metric = "LogLoss",
                   maximize = F,
                   trControl = ctrl_svm)
model_svm1$time <- proc.time() - ptm4

# Support Vector Machine object of class "ksvm" 
# 
# SV type: C-svc  (classification) 
# parameter : cost C = 0.25 
# 
# Gaussian Radial Basis kernel function. 
# Hyperparameter : sigma =  0.032565469406433 
# 
# Number of Support Vectors : 28086 
# 
# Objective Function Value : -304.1934 -238.7122 -162.779 -77.3837 -443.987 -378.7545 -525.7478 -564.8686 -2868.191 -935.4533 -143.0001 -348.3028 -599.1732 -359.0867 -365.3847 -844.5325 -109.4118 -267.5302 -558.3517 -281.2523 -268.0586 -81.6969 -227.7391 -324.6048 -165.7275 -180.2174 -72.4326 -79.6111 -77.1867 -83.5929 -496.4971 -823.9929 -520.9563 -468.5866 -335.0753 -557.9852 
# Training error : 0.238572 
Probability model included. 