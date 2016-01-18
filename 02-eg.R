##source: https://github.com/eugeneyan/kaggle_otto.git
library(doParallel)
detectCores()
registerDoParallel(detectCores() - 1) 
getDoParWorkers()

ctrl_rf <- trainControl(method = 'cv', 
                        number = 5, 
                        verboseIter = T, 
                        classProbs = T, 
                        summaryFunction = LogLossSummary)

grid_rf <- expand.grid(mtry = c(6, 9, 12))

model_rf <- train(target ~ ., data = trainset_59_feat, 
                  method = 'rf', 
                  metric = 'LogLoss', 
                  maximize = F,
                  tuneGrid = grid_rf, 
                  trControl = ctrl_rf, 
                  ntree = 500)

# create predictions
model_rf_pred <- as.matrix(predict(model_rf, newdata = testset, type = 'prob'))

# compute log.loss  
checkLogLoss(model_rf, model_rf_pred)  # logloss = 0.56234


ctrl_gbm <- trainControl(method = 'cv', 
                         number = 5, 
                         verboseIter = T, 
                         classProbs = T, 
                         summaryFunction = LogLossSummary)

grid_gbm <- expand.grid(interaction.depth = 10,
                        n.trees = (2:100) * 50,
                        shrinkage = 0.005)

model_gbm <- train(target ~ ., data = trainset_59_feat, 
                   method = 'gbm', 
                   distribution = 'multinomial', 
                   metric = 'LogLoss', 
                   maximize = F, 
                   tuneGrid = grid_gbm, 
                   trControl = ctrl_gbm,
                   n.minobsinnode = 4, 
                   bag.fraction = 0.9)

checkLogLoss(model_gbm, testset)  # log.loss = 0.509993


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
model_rf <- randomForest(target ~ ., data = trainset, 
                         mtry = 9, 
                         nodesize = 5, 
                         ntree = 500, 
                         keep.forest = T, 
                         importance = T)

# rf variable importance
imp_rf <- importance(model_rf, scale = T)
varImpPlot(model_rf, n.var = 20)  # top 20 most impt variables

# extract 20 most important rf variables
imp_rf <- as.data.frame(importance(rf.fit, type = 1))
imp_rf$Vars <- row.names(imp_rf)
imp_rf_20 <- imp_rf[order(-imp_rf$MeanDecreaseAccuracy),][1:20,]$Vars


# small gbm
grid_gbm <- expand.grid(interaction.depth = 10,
                        n.trees = 100,
                        shrinkage = 0.01)

model_gbm <- train(target ~., data = trainset, 
                   method = "gbm", 
                   distribution = 'multinomial', 
                   metric = 'LogLoss', 
                   maximize = F, 
                   tuneGrid = grid_gbm, 
                   trControl = ctrl,
                   n.minobsinnode = 4, 
                   bag.fraction = 0.9)
varImp(gbm.fit, scale = T)
imp_gbm <- data.frame(varImp(model_gbm)$importance)
imp_gbm$Vars <- row.names(imp_gbm)
imp_gbm_20 <- imp_gbm[order(-imp_gbm$Overall),][1:20,]$Vars

# combine top features identified by rf and gbm
feat_top <- unique(c(imp_gbm_20, imp_rf_20))


# small cforest (unusable as it takes very long to calculate conditional importance)
model_cf <- cforest(target ~., data = trainset, 
                  control = cforest_unbiased(mtry = 9, ntree = 200))
imp_cf <- varimp(cf.fit, conditional = F, threshold = 0.8)




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