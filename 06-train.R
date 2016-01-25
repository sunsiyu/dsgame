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


ctrl <- trainControl(method = "repeatedcv", 
                     number = 10,
                     repeats = 3)
# for whatever reason, outputs NaN for multi-class
model_nb <- train(Class ~ ., data = trainset1,
                  method = "nb", 
                  trControl = ctrl)

model_elm <- train(x = trainset[, -ncol(trainset)],
                  y = labels,
                  method = "elm", 
                  trControl = ctrl)

model_rf <- train(x = trainset[, -ncol(trainset)],
                  y = labels,
                  method = "rf", 
                  trControl = trainControl("oob"))

### svm
ctrl_svm <- trainControl(method = "cv", 
                         number = 5,
                         classProbs = T, 
                         savePredictions = T, 
                         verboseIter = T,
                         summaryFunction = LogLossSummary)
ptm <- proc.time()
model_svm = train(class ~ ., 
                  data = trainset, 
                  method = "svmRadial", 
                  preProcess = c("center", "scale"),
                  metric = "LogLoss",
                  maximize = F,
                  trControl = ctrl_svm)
model_svm$time <- proc.time() - ptm

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
# Objective Function Value : -304.1934 -238.7122 -162.779 -77.3837 -443.987 
# -378.7545 -525.7478 -564.8686 -2868.191 -935.4533 -143.0001 -348.3028 -599.1732 
# -359.0867 -365.3847 -844.5325 -109.4118 -267.5302 -558.3517 -281.2523 -268.0586 
# -81.6969 -227.7391 -324.6048 -165.7275 -180.2174 -72.4326 -79.6111 -77.1867 
# -83.5929 -496.4971 -823.9929 -520.9563 -468.5866 -335.0753 -557.9852 
# 
# Training error : 0.238572 
# Probability model included. 


### nn
ctrl_nn <- trainControl(method = "cv", 
                        number = 5, 
                        verboseIter = T,
                        savePredictions = T,
                        classProbs = T,
                        summaryFunction = LogLossSummary)

grid_nn <- expand.grid(size = c(2, 3, 4, 5), decay = c(1e-2, 2e-2, 5e-2))

ptm_nn <- proc.time()

model_nn <- train(class ~., data = trainset, 
                  method = "nnet",
                  metric = "LogLoss", 
                  maximize = F,
                  preProcess = c("scale", "center"), 
                  tuneGrid = grid_nn,
                  trControl = ctrl_nn)

model_nn$time <- proc.time() - ptm_nn


## only disappointed to found that with fewer features limited to 50, damage the result by around 0.3
# > model_nn
# Neural Network 
# 
# 39611 samples
# 45 predictor
# 9 classes: 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9' 
# 
# Pre-processing: scaled (45), centered (45) 
# Resampling: Cross-Validated (5 fold) 
# Summary of sample sizes: 31690, 31689, 31689, 31689, 31687 
# Resampling results across tuning parameters:
#   
#   size  decay  LogLoss   LogLoss SD
# 2     0.01   1.448749  0.03061474
# 2     0.02   1.461972  0.03709139
# 2     0.05   1.430667  0.02649941
# 3     0.01   1.385401  0.01772194
# 3     0.02   1.418925  0.02630946
# 3     0.05   1.394464  0.03790382
# 4     0.01   1.339638  0.01846866
# 4     0.02   1.340224  0.01251382
# 4     0.05   1.326989  0.02298521
# 5     0.01   1.312397  0.01023066
# 5     0.02   1.304844  0.02700196
# 5     0.05   1.308214  0.01214169
# 
# LogLoss was used to select the optimal model using  the smallest value.
# The final values used for the model were size = 5 and decay = 0.02. 



### xgboost
library(xgboost)
# Xgboost manages only numeric vectors.
y <- gsub('Class_', replacement = "", x = trainset[, ncol(trainset)])
y <- as.integer(y)
y <- y - 1
trainset <- as.matrix(trainset[, -ncol(trainset)])

ctrl_xgb <- list("objective" = "multi:softprob",
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
model_xgb <- xgb.cv(param = ctrl_xgb, 
                    data = trainset, 
                    label = y, 
                    nfold = 5, 
                    nrounds = 10000)
# check best iteration
cv_min <- min(model_xgb$test.mlogloss.mean)
cv_min_rounds <- which(model_xgb$test.mlogloss.mean == min(model_xgb$test.mlogloss.mean))  
# min = 0.517568 at nrounds 4831
cv_rounds <- round(mean(which(model_xgb$test.mlogloss.mean == min(model_xgb$test.mlogloss.mean))))
# 4831

# fit model on training set
xgb_fit <- xgboost(param = ctrl_xgb, 
                   data = trainset, 
                   label = y, 
                   nrounds = cv_rounds)

checkLogLoss(xgb.fit, train.x[-in.train, ], train[-in.train, ])  
# log.loss = 0.4517306 (improvement of 0.0006185)


### RESULT
# > model_xgb
# train.mlogloss.mean train.mlogloss.std test.mlogloss.mean test.mlogloss.std
# 1:            2.184209           0.000111           2.184889          0.000205
# 2:            2.171634           0.000326           2.173037          0.000264
# 3:            2.159308           0.000338           2.161483          0.000297
# 4:            2.146946           0.000453           2.149896          0.000612
# 5:            2.134884           0.000606           2.138541          0.000689
# ---                                                                            
#   9996:            0.121460           0.000379           0.520156          0.006371
# 9997:            0.121455           0.000379           0.520156          0.006371
# 9998:            0.121453           0.000379           0.520156          0.006371
# 9999:            0.121450           0.000381           0.520158          0.006371
# 10000:            0.121444           0.000380           0.520159          0.006373