# ===================
# LOAD PACKAGES
# ===================
library(ggplot2)
library(dplyr)
library(DMwR)
library(tree)
library(rpart)
library(caret)
library(randomForest)
library(adabag)
library(gbm)
library(doParallel)
# library(doMC)

# ===================
# LOAD DATA & SPLIT
# ===================
set.seed(114)
train <- read.table(dir(".", "train.csv", recursive = T), header=T, sep=",")
train <- train[, -1]  # remove id
intrain <- createDataPartition(train$target, p = 0.8, list = FALSE)
trainset <- train[intrain, ]
testset <- train[-intrain, ]


# Basic Decision Tree 2
ptm2 <- proc.time()
model_rpart <- rpart(target ~ ., data = trainset, method = "class")
model_rpart$time <- proc.time() - ptm2

# Adabag
ptm3 <- proc.time()
model_adabag <- bagging(target ~ ., data = trainset, mfinal=50)
model_adabag$time <- proc.time() - ptm3

# Random Forest
# Use the tuneRF function to determine an ideal value for the mtry parameter
# The ideal mtry value was found to be 8
mtry <- tuneRF(trainset, labels, 
               mtryStart=1, 
               ntreeTry=50, 
               stepFactor=2, 
               improve=0.05,
               trace=TRUE, 
               plot=TRUE, 
               doBest=FALSE)
ptm4 <- proc.time()
model_rf <- randomForest(Class ~ ., data = trainset, 
                         importance=TRUE, 
                         ntree=100, 
                         mtry=8)
model_rf$time <- proc.time() - ptm4
model_rf_pred <- predict(model_rf, testset, type="prob")

# Boosting
ptm5 <- proc.time()
model_gbm <- gbm(Class ~ ., data = trainset, 
                 distribution="multinomial", 
                 n.trees=1000, 
                 shrinkage=0.05, 
                 interaction.depth=12, 
                 cv.folds=2)
model_gbm$time <- proc.time() - ptm5
trees <- gbm.perf(model_gbm)


# knn + doParallel
indx <- createFolds(trainset$Class, returnTrain = TRUE)
ctrl <- trainControl(method = "LGOCV",  
                     classProbs = TRUE,
                     index = indx,
                     savePredictions = TRUE)
cl <- makeCluster(3)
registerDoParallel(cl)
#Running the knn over a range of k values 
knn.time1 <- system.time(model_knn <- train(x = trainset,
                                            y = labels,
                                            method = "knn",
                                            preProc = c("center", "scale"),
                                            tuneGrid = data.frame(k =c(4*(0:5)+1)),
                                            trControl = ctrl))
#ROC metric did not run so accuracy was defaulted to. Turns out ROC is only
#good for two class classification problems
model_knn$pred <- merge(model_knn$pred,  model_knn$bestTune)
confusionMatrix(model_knn, norm = "none")
plot(model_knn, metric="Accuracy")
model_knn_pred <- predict(model_knn, newdata = testset)
#Calculating model performance
postResample(pred = model_knn_pred, obs = testset_labels)


# random forest + doMC
registerDoMC(cores = 4)
ctrl1 <- trainControl(method = "repeatedcv", 
                     repeats = 5, 
                     classProbs = TRUE)
ctrl2 <- trainControl(method = "boot", 
                      classProbs = TRUE)

rfGrid <-  expand.grid(mtry = c(5,9,18))

model_rf2 <- train(x = trainset, 
                   y = labels, 
                   method="rf",
                   trControl = ctrl1,
                   allowParallel=TRUE, 
                   tuneGrid = rfGrid)


# pls
model_pls <- train(x = trainset, 
                   y = labels,
                   method = "pls",
                   tuneLength = 15,
                   trControl = ctrl1,
                   preProc = c("center", "scale"))



# rf








#List of frequency tables for easy access
tables <- lapply(train[, 2:94], table)
#List of histogram plots to easily see skew of all variables.
plots <- lapply(train[, 2:94], qplot)
plot(model_tree, main = "tree model using pkg tree")
par(xpd=TRUE)
plot(model_rpart, compress=TRUE, main = "rpart")

# Create a dotchart of variable/feature importance as measured by a Random Forest
varImpPlot(model_rf)


model_tree_pred <- predict(model_tree, testset, type="class")
table(model_tree_pred, testset$Class)
model_tree_pred$error <- 1-(sum(model_tree_pred==testset$Class)/length(testset$Class))

model_rpart_pred <- predict(model_rpart, testset, type="class")
table(model_rpart_pred, testset$Class)
model_rpart$error <- 1-(sum(model_rpart_pred==testset$Class)/length(testset$Class))

model_adabag_pred <- predict(model_adabag, testset, newmfinal=50)
table(as.factor(model_adabag_pred$class),testset$Class)


model_rf_pred <- predict(model_rf, testset, type="response")
table(model_rf_pred,stest$target)
model_rf$error <- 1-(sum(model_rf_pred==testset$Class)/length(testset$Class))
model_rf$error



################################################################################








# summaryFunction
LogLoss <- function(actual, predicted, eps=0.00001) {
  predicted <- pmin(pmax(predicted, eps), 1-eps)
  -sum(actual*log(predicted))
}

# 
# Otto classification challenge from Kaggle
# The first r script, productclassifcation uses very basic random forest and obtained a score of 0.64283 (mediocre, about top 45%)
# The second, productclassifiationXGB, was a preliminary attempt at using XGBoost to get a better score. Due to likely formatting issues, it obtained a score of 2.73918 (very bad, bottom 20%)
# The third, productclassificationXGB2, was an attempt at using XGBoost for classification and got my best score of 0.48729 (decent, about top 30% of kagglers in the competition)

#simple product classification using random forest
library(readr)
library(mclust)
results <- data.frame(id=testData$id, Class_1=NA, Class_2=NA, Class_3=NA, Class_4=NA, Class_5=NA, Class_6=NA, Class_7=NA, Class_8=NA, Class_9=NA)

randomForest <- randomForest(trainData[,c(-1,-95)], as.factor(trainData$target), ntree=25, importance=TRUE)
results[,2:10] <- (predict(randomForest, testData[,-1], type="prob")+0.01)/1.09

imp <- importance(randomForest, type=1)
featureImportance <- data.frame(Feature=row.names(imp), Importance=imp[,1])

p <- ggplot(featureImportance, aes(x=reorder(Feature, Importance), y=Importance)) +
  geom_bar(stat="identity", fill="#53cfff") +
  coord_flip() + 
  theme_light(base_size=20) +
  xlab("Importance") +
  ylab("") + 
  ggtitle("Random Forest Feature Importance\n") +
  theme(plot.title=element_text(size=18))

ggsave(file = "feature_importance.png", p, path = "~/Dropbox/Kaggle/Otto", height=20, width=8, units="in")


#### using gradient boosting algorithm
# Install XGBoost
devtools::install_github('dmlc/xgboost',subdir='R-package')

# Packages and dataset
require(devtools)
require(xgboost)
require(methods)
require(data.table)
require(magrittr)
require(ggplot2)
require(DiagrammeR)
require(Ckmeans.1d.dp)

# Convert from classes to numbers for XGBoost support
y <- train[, nameLastCol, with = F][[1]] %>% gsub('Class_','',.) %>% {as.integer(.) -1}
# Display the first 5 levels
y[1:5]

# Delete label column or it will be used in prediction
train[, nameLastCol:=NULL, with = F]

# Convert data tables into numeric Matrices, also for XGBoost support
trainMatrix <- train[,lapply(.SD,as.numeric)] %>% as.matrix
testMatrix <- test[,lapply(.SD,as.numeric)] %>% as.matrix

# Train the model
numberOfClasses <- max(y) + 1
numberOfClasses
y

param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = numberOfClasses)

cv.nround <- 5
cv.nfold <- 3

bst.cv = xgb.cv(param=param, data = trainMatrix, label = y, 
                nfold = cv.nfold, nrounds = cv.nround)

nround = 50
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = numberOfClasses)
bst = xgboost(param=param, data = trainMatrix, label = y, nrounds=nround)

# View the model
model <- xgb.dump(bst, with.stats = T)
model[1:10]

# Feature importance/selection, find the 10 most important features
# Get the feature real names
names <- dimnames(trainMatrix)[[2]]

# Compute feature importance matrix
importance_matrix <- xgb.importance(names, model = bst)

# Nice graph
xgb.plot.importance(importance_matrix[1:10,])

# Interaction between features
xgb.plot.tree(feature_names = names, model = bst, n_first_tree = 2, width = 3000, height = 1600)


# Predict
pred = predict(bst, testMatrix[, -1])
pred = matrix(pred,9,length(pred)/9)
pred = t(pred)
pred = format(pred, digits=2,scientific=F) # shrink the size of submission
pred = data.frame(1:nrow(pred),pred)
names(pred) = c('id', paste0('Class_',1:9))

randomForest <- randomForest(trainData[,c(-1,-95)], as.factor(trainData$target), ntree=25, importance=TRUE)
results[,2:10] <- (predict(randomForest, testData[,-1], type="prob")+0.01)/1.09


require(xgboost)
require(methods)

#this is the final column (selection)
y = train[,ncol(train)]
y = gsub('Class_','',y)
y = as.integer(y)-1 #xgboost take features in [0,numOfClass)

x = rbind(train[,-ncol(train)],test)
x = as.matrix(x)
x = matrix(as.numeric(x),nrow(x),ncol(x))
trind = 1:length(y)
teind = (nrow(train)+1):nrow(x)

# Set necessary parameter
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 9,
              "nthread" = 8)

# Run Cross Valication
cv.nround = 80
bst.cv = xgb.cv(param=param, data = x[trind,], label = y, 
                nfold = 3, nrounds=cv.nround)

# Train the model
nround = 80
bst = xgboost(param=param, data = x[trind,], label = y, nrounds=nround)

# Make prediction
pred = predict(bst,x[teind,])
pred = matrix(pred,9,length(pred)/9)
pred = t(pred)

# Output submission
pred = format(pred, digits=2,scientific=F) # shrink the size of submission
pred = data.frame(1:nrow(pred),pred)
names(pred) = c('id', paste0('Class_',1:9))
write.csv(pred,file='~/Dropbox/Kaggle/Otto/anotherxgb.csv', quote=FALSE,row.names=FALSE)

predold <- read.csv('submission.csv',header=TRUE,stringsAsFactors = F)
View(predold)



############# another guy
library(doParallel)
cl = makeCluster(4)
registerDoParallel(cl)

mcLogLoss <- function (data,
                       lev = NULL,
                       model = NULL) {
  
  if (!all(levels(data[, "pred"]) == levels(data[, "obs"])))
    stop("levels of observed and predicted data do not match")
  
  LogLoss <- function(actual, pred, err=1e-15) {
    pred[pred < err] <- err
    pred[pred > 1 - err] <- 1 - err
    -1/nrow(actual)*(sum(actual*log(pred)))
  }
  
  dtest <- dummyVars(~obs, data=data, levelsOnly=TRUE)
  actualClasses <- predict(dtest, data[,-1])
  
  out <- LogLoss(actualClasses, data[,-c(1:2)])  
  names(out) <- "mcLogLoss"
  out
}

mtryGrid = expand.grid(mtry = c(3,5,7,9))

fitControl <- trainControl(method = "cv",
                           number = 5,
                           classProbs = TRUE,
                           summaryFunction = mcLogLoss)

rfTune<- train(x = data,
               y = labels,
               method = "rf",
               trControl = fitControl,
               metric = "mcLogLoss",
               ntree = 1000,
               tuneGrid = mtryGrid, 
               maximize = FALSE,
               importance = TRUE)

# Random Forest 
# 
# 61878 samples
# 103 predictor
# 9 classes: 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9' 
# 
# No pre-processing
# Resampling: Cross-Validated (5 fold) 
# 
# Summary of sample sizes: 49503, 49501, 49504, 49502, 49502 
# 
# Resampling results across tuning parameters:
#   
#   mtry  mcLogLoss  mcLogLoss SD
# 3     0.6832063  0.001371156 
# 5     0.6280732  0.001462273 
# 7     0.6019547  0.002529347 
# 9     0.5862516  0.002805598 
# 
# mcLogLoss was used to select the optimal model using  the smallest value.
# The final value used for the model was mtry = 9. 


data = read.csv('engineered1train.csv')
class = data$class

xgbtrain = data[,2:105]
xgbtrain = xgbtrain[,-104]


class = gsub('Class_','',class)
class = as.numeric(class) - 1

param <- list('objective' = 'multi:softprob',
              'eval_metric' = 'mlogloss',
              'num_class' = 9,
              'nthread' = 4,
              'max.depth' = 7)

xgbfinalblendMod = xgboost(param=param, 
                           data = xgbtrain, 
                           label = class, 
                           nrounds=494, 
                           eta=0.08, 
                           colsample.bytree=0.45)


xgb.save(xgbfinalblendMod, 'xgbfinalblendmodel')

xgblendfinal_pred = predict(xgbfinalblendMod, xgbtest)
xgblendfinal_pred = matrix(xgblendfinal_pred,9,length(xgblendfinal_pred)/9)
xgblendfinal_pred = t(xgblendfinal_pred)


################################################H2O####################################################
library(h2o)
localH2O <- h2o.init(nthread=4, Xmx='8g')

class = data$class
train = data[,2:105]
train = train[,-104]
train$class = class

test = datatest[,2:105]
test = test[,-104]

for(i in 1:(ncol(train)-1)){
  train[,i] <- as.numeric(train[,i])
  train[,i] <- sqrt(train[,i]+(3/8))
}

for(i in 1:(ncol(test))){
  test[,i] <- as.numeric(test[,i])
  test[,i] <- sqrt(test[,i]+(3/8))
}


train.hex <- as.h2o(localH2O,train)
test.hex <- as.h2o(localH2O,test)

predictors <- 1:(ncol(train.hex)-1)
response = ncol(train.hex)


submission <- read.csv("sampleSubmission.csv")
submission[,2:10] <- 0

for(i in 1:20){
  print(i)
  model <- h2o.deeplearning(x=predictors,
                            y=response,
                            data=train.hex,
                            classification=T,
                            activation="RectifierWithDropout",
                            hidden=c(1024,512,256),
                            hidden_dropout_ratio=c(0.5,0.5,0.5),
                            input_dropout_ratio=0.05,
                            epochs=100,
                            l1=1e-5,
                            l2=1e-5,
                            rho=0.99,
                            epsilon=1e-8,
                            train_samples_per_iteration=2000,
                            max_w2=10,
                            seed=1)
  
  submission[,2:10] <- submission[,2:10] + as.data.frame(h2o.predict(model,test.hex))[,2:10]
  print(i)
  write.csv(submission,file="submission.csv",row.names=FALSE) 
}   
submission = read.csv('submission.csv')


submission1 = submission1[,-1]
subSums = rowSums(submission1)
submissionNormed1 = sweep(as.matrix(submission1), 1, subSums, `/`)
colnames(submissionNormed) = paste0(rep('Class_',9),1:9)
submissionNormed1 = as.data.frame(submissionNormed1)
submissionNormed$obs = test$target
LogLoss(submissionNormed)




LogLoss = function(data, lev=NULL, model=NULL){
  pred = data[,-which(colnames(data) == 'obs')]
  eps = 1e-15
  predsnormed = do.call(cbind, lapply(pred, function(x) sapply(x, function(y) max(min(y, 1-eps), eps))))
  logProbs = log(as.matrix(predsnormed))  
  log1minus = log(1-as.matrix(predsnormed))  
  out = rep(NA, nrow(data))
  for(i in 1:length(data$obs)){
    colidx = which(data$obs[i] == colnames(logProbs))
    out[i] = sum(logProbs[i,colidx], log1minus[i,-colidx])
  }
  return(-sum(out)/length(out))
}

##Find overall model weights##

#first i is 1:10 0.7866895

#48% h2o / 52% xgboost 0.7866227
for(i in 40:60){
  weighted_pred = (i*submissionNormed + (100-i)*xgblend_pred)/100
  weighted_pred$obs = paste0('Class_', as.character((testlabels+1)))
  ll = LogLoss(weighted_pred)
  print(i)
  print(ll)
}

weighted_pred = (48*submissionNormed + 52*xgblend_pred)/100
weighted_pred$obs = paste0('Class_', as.character((testlabels+1)))

##Dive deeper beyond model averages

for(j in 1:9){
  for(i in 1:10){
    weighted_pred1 = weighted_pred
    print(paste0('h2o weight = ', i))
    print(paste0('column',j))
    weighted_pred1[,j] = (i*submissionNormed[,j]+(10-i)*xgblend_pred[,j])/10
    ll = LogLoss(weighted_pred1)
    print(ll)
  }
}


#class 1; h20 = 8, xgboost = 2, 0.786466
#class 2; h20 = 5, xgboost = 5, 0.7865928
#class 3; h2o = 4, xgboost = 6, 0.7864068
#class 4; h20 = 3, xgboost = 7, 0.7862879
#class 5; h20 = 4, xgboost = 6, 0.786622
#class 6; h20 = 5, xgboost = 5, 0.7866182
#class 7; h20 = 5, xgboost = 5, 0.7866383
#class 8; h20 = 5, xgboost = 5, 0.7866125
#class 9; h20 = 5, xgboost = 5, 0.7866207

weighted_pred2 = weighted_pred1
weighted_preds3[,1] = (8*subNorm[,1] + 2*xgblendfinal_pred[,1])/10
weighted_preds3[,3] = (4*subNorm[,3] + 6*xgblendfinal_pred[,3])/10
weighted_preds3[,4] = (3*subNorm[,4] + 7*xgblendfinal_pred[,4])/10
weighted_preds3[,5] = (4*subNorm[,5] + 6*xgblendfinal_pred[,5])/10

LogLoss(weighted_pred2) #0.7859146

weighted_pred2 = as.matrix(weighted_pred2)
weighted_pred2 = matrix(as.numeric(weighted_pred2), nrow = nrow(weighted_pred2), ncol=ncol(weighted_pred2))
weighted_pred2 = weighted_pred2[,-10]

param <- list('objective' = 'multi:softprob',
              'eval_metric' = 'mlogloss',
              'num_class' = 9,
              'nthread' = 4,
              'max.depth' = 2)

xgbmod.cv = xgb.cv(param=param, data = weighted_pred2, label = testlabels, 
                   nfold = 3, nrounds=1200, eta=0.1)

#max.depth = 1, nround = 52, eta=0.3, colsample.bytree=1, test ll= 0.455617
#max.depth = 1, nround = 147, eta=0.1, colsample.bytree=1, test ll= 0.454446
#max.depth = 1, nround = 236, eta=0.1, colsample.bytree=0.5, test ll= 0.453400
#max.depth = 1, nround = 357, eta=0.1, colsample.bytree=0.25, test ll= 0.458217

#max.depth = 1, nround = 47, eta=0.08, colsample.bytree=0.45, test ll= 0.469667

#max.depth = 2, nround = 29, eta=0.3, colsample.bytree=1, test ll= 0.455414
#max.depth = 2, nround = 87, eta=0.1, colsample.bytree=1, test ll= 0.453182**
#max.depth = 2, nround = 155, eta=0.1, colsample.bytree=0.5, test ll= 0.459024
#max.depth = 2, nround = 207, eta=0.1, colsample.bytree=0.25, test ll= 0.465074

#max.depth = 3, nround = 25, eta=0.3, colsample.bytree=1, test ll= 0.465275
#max.depth = 3, nround = 75, eta=0.1, colsample.bytree=1, test ll= 0.459041
#max.depth = 3, nround = 114, eta=0.1, colsample.bytree=0.5, test ll= 0.464345
#max.depth = 3, nround = 188, eta=0.1, colsample.bytree=0.25, test ll= 0.474762

#max.depth = 4, nround = 21, eta=0.3, colsample.bytree=1, test ll= 0.468915
#max.depth = 4, nround = 69, eta=0.1, colsample.bytree=1, test ll= 0.472718
#max.depth = 4, nround = 100, eta=0.1, colsample.bytree=0.5, test ll= 0.468746
#max.depth = 4, nround = 171, eta=0.1, colsample.bytree=0.25, test ll= 0.484421

#max.depth = 5, nround = 21, eta=0.3, colsample.bytree=1, test ll= 0.478406
#max.depth = 5, nround = 67, eta=0.1, colsample.bytree=1, test ll= 0.478422
#max.depth = 5, nround = 89, eta=0.1, colsample.bytree=0.5, test ll= 0.477553
#max.depth = 5, nround = 155, eta=0.1, colsample.bytree=0.25, test ll= 0.487217

#max.depth = 6, nround = 21, eta=0.3, colsample.bytree=1, test ll= 0.490719
#max.depth = 6, nround = 68, eta=0.1, colsample.bytree=1, test ll= 0.485659
#max.depth = 6, nround = 82, eta=0.1, colsample.bytree=0.5, test ll= 0.481521
#max.depth = 6, nround = 137, eta=0.1, colsample.bytree=0.25, test ll= 0.498963

#max.depth = 7 not competitive

xgbfinalblendMod = xgboost(param=param, data = weighted_pred2, label = class, nrounds=87, eta=0.1)

#######################testing ensembles####################################
test = read.csv('engineered1test.csv')
xgbtest = test[,2:105]
xgbtest = xgbtest[,-104]
xgbtest = as.matrix(xgbtest)
xgbtest = matrix(as.numeric(xgbtest), nrow = nrow(xgbtest), ncol=ncol(xgbtest))

xgbpred_sub = predict(xgblendMod, xgbtest)




############### too long
#For Otto Kaggle Competition

#exploring features
library(plyr)
library(dplyr)
library(ggplot2)
library(reshape2)
train$target = NULL
eventrates = ddply(train, .(targets), function(x) colSums(x!=0)/nrow(x))
classSums = rowSums(eventrates[,-1])
eventrates.mat = as.matrix(eventrates[,-1])
normalized = sweep(eventrates.mat, 1, classSums, `/`)
normalized = as.data.frame(cbind(targets = seq(1,9,1), normalized))
eventrate_m = melt(eventrates, id='targets')classSu
normalized_m = melt(normalized, id='targets')

#event rate and normalized event-rate by class
ggplot(eventrate_m, aes(x=variable, y=value, fill=targets, label=variable))+
  ylab('Event-rate')+geom_bar(stat='identity',position='dodge')+xlab('Feature')+
  theme_bw()+theme(axis.text.x=element_text(angle=45, hjust=1))+facet_wrap(~targets)+
  geom_text()

ggplot(normalized_m, aes(x=variable, y=value, fill=factor(targets), label=variable))+
  ylab('Normalized Event-rate')+geom_bar(stat='identity',position='dodge')+xlab('Feature')+
  theme_bw()+theme(axis.text.x=element_text(angle=45, hjust=1))+facet_wrap(~targets)+
  geom_text()

#plot feature ranges by class
library(reshape2)
library(ggplot2)
train$id = seq(1, nrow(train), 1)
train$target = targets
train_m = melt(train, id=c('target','id'), class = variable)
feat.ranges = ddply(train_m, .(target, variable), function(x) range(x$value))

ggplot(feat.ranges, aes(x=variable, y=V2, group=target, fill=target, label=variable))+xlab('')+
  geom_bar(stat='identity')+facet_wrap(~target)+ylab('Feature Range')+theme_bw()+geom_text()+
  theme(axis.text.x=element_blank())

#plot typical values (mean & median)
feat.centers = ddply(train_m, .(target, variable), summarize, Mean = mean(value), Median = median(value))
ggplot(feat.centers, aes(x=variable, y=Mean, group=target, fill=target, label=variable))+xlab('')+
  geom_bar(stat='identity')+facet_wrap(~target)+ylab('Feature Mean')+theme_bw()+geom_text()+
  theme(axis.text.x=element_blank())

ggplot(feat.centers, aes(x=variable, y=Median, group=target, fill=target, label=variable))+xlab('')+
  geom_bar(stat='identity')+facet_wrap(~target)+ylab('Feature Median')+theme_bw()+geom_text()+
  theme(axis.text.x=element_blank())

#heat map with all features
library(tidyr)
library(gplots)
Col.Scale = colorRampPalette(colors=c('blue','white','red'))(5)
heat.Med = spread(feat.centers[,-3],variable, Median)[,-1]
heatmap.2(data.matrix(heat.Med), Rowv=TRUE, Colv=FALSE, scale='column', trace='none', 
          col=Col.Scale, margins = c(3,5), ylab='Class', xlab='Feature Median')

#replot heatmap with only features w/ median > 0
idx = which(colSums(heat.Med)!=0)
heat.Med.small = select(heat.Med, idx)
dev.off()
heatmap.2(data.matrix(heat.Med.small), Rowv=TRUE, Colv=FALSE, scale='column', trace='none', 
          col=Col.Scale, margins = c(3,5), ylab='Class', xlab='Feature Median')


#Build LogLoss evaluation metric used by Kaggle
LogLoss = function(data, lev=NULL, model=NULL){
  pred = data[,-which(colnames(data) == 'obs')]
  eps = 1e-15
  predsnormed = do.call(cbind, lapply(pred, function(x) sapply(x, function(y) max(min(y, 1-eps), eps))))
  logProbs = log(as.matrix(predsnormed))  
  log1minus = log(1-as.matrix(predsnormed))  
  out = rep(NA, nrow(data))
  for(i in 1:length(data$obs)){
    colidx = which(data$obs[i] == colnames(logProbs))
    out[i] = sum(logProbs[i,colidx], log1minus[i,-colidx])
  }
  return(-sum(out)/length(out))
}

##Unable to use caret for tuning parameter search and model eval due to memory requirements/burden/leakage##
##single gbm object ~1gb, only have 8gb##
##Building parallelized function for model evaluation and tuning parameter selection##


#register parallel backend
library(doParallel)
cl = makeCluster(4)
registerDoParallel(cl)


#no cv using logloss fxn, just holdout test set for model evaluation
#initial gbm tuning parameters
trees = c(100,150,200,250,300)
depth = c(3,5,7)

#tune over grid of parameters
tune = foreach(i = trees, .packages='gbm') %:% 
  foreach(j = depth, .combine = 'rbind') %dopar% {
    gbmMod = gbm(target~., data= train2[,-1], distribution='multinomial', n.trees=i, 
                 interaction.depth=j, shrinkage = 0.01, n.cores=3)
    gbmPred = predict(gbmMod, test[,-1], n.trees=i, type='response')
    gbmPreddf = as.data.frame(gbmPred[,,1])
    gbmPreddf$obs = test$target
    LogLoss(gbmPreddf)
  }

#first model submission with best parameters chosen, unsurprisingly the are max trees (300) & depth (7)
finalMod = gbm(target~., data= train[,-1], distribution='multinomial', n.trees=300, interaction.depth=7, shrinkage = 0.01, n.cores=3)
##Results: first submission score 0.70854 - beats uniform probability:2.19, and rf benchmarks:1.50
##best logloss was with highest number of trees and deepest depth, going to continue increasing trees/depth
##features 1,3,6,10,12,13,21,27,28,29,31,37,46,49,51,52,61,63,65,66,73,74,80,81,82,87,89 have zero Var Imp.


#new tuning parameters
trees = c(350,400,450,500,550)
depth = c(9,11)

#second search in parameter space
tune2 = foreach(i = trees, .packages='gbm') %:% 
  foreach(j = depth, .combine = 'rbind') %dopar% {
    gbmMod = gbm(target~., data= train2[,-1], distribution='multinomial', n.trees=i, 
                 interaction.depth=j, shrinkage = 0.01, n.cores=3)
    gbmPred = predict(gbmMod, test[,-1], n.trees=i, type='response')
    gbmPreddf = as.data.frame(gbmPred[,,1])
    gbmPreddf$obs = test$target
    LogLoss(gbmPreddf)
  }

finalMod2 = gbm(target~., data= train[,-1], distribution='multinomial', n.trees=550, 
                interaction.depth=11, shrinkage = 0.01, n.cores=4)

#make submissions, only second output shown
submit2 = predict(finalMod2, test[,-1], n.trees=550, type='response')
submit2 = as.data.frame(submit1[,,1])
write.csv(submit2, 'submit2.csv')
finalMod2.varImp = summary(finalMod2)
write.csv(finalMod2.varImp, 'submit2varImp.csv')
##Results: second submission score 0.59317
##best logloss was with highest number of trees and deepest depth, going to continue increasing trees/depth
##6,12,21,27,28,31,37,52,63,82 zero var imp
##gbm quickly becoming too computationally complex for my machine, switched to xgboost.

stopCluster(cl)

##xgboost - tuning. Tuned depth, then n.trees.

library(xgboost)
library(methods)


train$target = gsub('Class_', '', train$target)
class = as.numeric(train$target) - 1
train$target = NULL

train = as.matrix(train)
train = matrix(as.numeric(train), nrow = nrow(train), ncol=ncol(train))
test = as.matrix(test)
test = matrix(as.numeric(test), nrow=nrow(test))

param <- list('objective' = 'multi:softprob',
              'eval_metric' = 'mlogloss',
              'num_class' = 9,
              'nthread' = 4,
              'max.depth' = 5)

xgbmod.cv = xgb.cv(param=param, data = train, label = class, 
                   nfold = 3, nrounds=200)

xgbfinalMod = xgboost(param=param, data = train, label = class, nrounds=53)

# Make prediction
submit4 = predict(xgbfinalMod,test)
submit4 = matrix(submit4,9,length(submit4)/9)
submit4 = t(submit4)
submit4 = as.data.frame(submit4)
submit4 = cbind(id = 1:nrow(submit4), submit4)
names(submit4) = c('id', paste0('Class_',1:9))
write.csv(submit4, file='submit4.csv', quote=FALSE,row.names=FALSE)

##submission result using max.depth = 6, nround = 50 was 0.50763
##submission results using max.depth = 10, nround = 57 was 0.47653

##cv 3-fold tuning
##max.depth = 6 0.524514
##max.depth = 7 gave 0.514282
##max.depth = 8 gave 0.508554
##max.depth = 9 gave 0.504817
##max.depth = 10, nrounds=50 gave 0.503991, nrounds = 57 gave 0.503269
##max.depth = 11 gave 0.510535 <- overfitting

importance_matrix <- xgb.importance(colnames(train), model = xgbfinalMod)
xgb.plot.importance(importance_matrix)

##Feature Engineering

#counts of all unique numbers per row
nums = unique(as.numeric(as.matrix(train))) #leave nums unchanged from train set

p = matrix(ncol=length(nums))
colnames(p) = nums
p = foreach(i = 1:nrow(train), .combine='rbind') %dopar% {
  sapply(nums, function(x) sum(x == train[i,]))
}


colnames(p) = sapply(nums, function(x) paste0('Num', x))

train = cbind(train, rsum = rowSums(train)) #rowsums
train = cbind(train, p)

write.csv(cbind(train, class), 'engineered1train.csv')

test = read.csv('ottotest.csv')
test = test[,-1]

l = matrix(ncol=length(nums))
colnames(l) = nums
l = foreach(i = 1:nrow(test), .combine='rbind') %dopar% {
  sapply(nums, function(x) sum(x == test[i,]))
}


colnames(l) = sapply(nums, function(x) paste0('Num', x))

test = cbind(test, rsum = rowSums(test)) #rowsums
test = cbind(test, l)

write.csv(test, 'engineered1test.csv')

stopCluster(cl)

##New Model w/ features

class = gsub('Class_','',class)
class = as.numeric(train$target) - 1

train = as.matrix(train)
train = matrix(as.numeric(train), nrow = nrow(train), ncol=ncol(train))
colnames(train) = colnames(test)
test = as.matrix(test)
test = matrix(as.numeric(test), nrow=nrow(test))
colnames(test) = colnames(train)

param <- list('objective' = 'multi:softprob',
              'eval_metric' = 'mlogloss',
              'num_class' = 9,
              'nthread' = 4,
              'max.depth' = 6)

xgbmod.cv = xgb.cv(param=param, data = train, label = class, 
                   nfold = 3, nrounds=130) #nround =56, max.depth=10, gives 0.494899

xgbfinalMod = xgboost(param=param, data = train, label = class, nrounds=104)

submit5 = predict(xgbfinalMod,test)
submit5 = matrix(submit5,9,length(submit5)/9)
submit5 = t(submit5)
submit5 = as.data.frame(submit5)
submit5 = cbind(id = 1:nrow(submit5), submit5)
names(submit5) = c('id', paste0('Class_',1:9))
write.csv(submit5, file='submit5.csv', quote=FALSE,row.names=FALSE)
#scored 0.47704 not better than previous score

#using ONLY most frequent feature counts
train = train[,1:105]
train = train[,-104]

#retune -
#max.depth = 11 gave nround(54) = 0.515250
#max.depth = 10 gave nround(53) = 0.508982
#max.depth = 9 gave nround(65) = 0.504933
#max.depth = 8 gave nround(74) = 0.506664 
#max.depth = 7 gave nround(104) = 0.499573 - submission score = 0.47365. new best (renamed this submit5.)
#max.depth = 6 gave nround(130) = 0.500334


#now tune with ALL new features
#retune - 
#max.depth = 10 gave nround(54) = 0.509774
#max.depth = 9 gave nround(72) = 0.505913
#max.depth = 8 gave nround(85) =  0.503530
#max.depth = 7 gave nround(104) = 0.499491 - submit6 score = 0.47365. exactly the same as submit5.
#max.depth = 6 gave nround(128) = 0.500250 (may benefit from more nrounds)

submit6 = predict(xgbfinalMod,test)
submit6 = matrix(submit6,9,length(submit6)/9)
submit6 = t(submit6)
submit6 = as.data.frame(submit6)
submit6 = cbind(id = 1:nrow(submit6), submit6)
names(submit6) = c('id', paste0('Class_',1:9))
write.csv(submit6, file='submit6.csv', quote=FALSE,row.names=FALSE)


xgb.save(xgbfinalMod, 'xgbfinalmodel')
xgb.load('xgbfinalmodel')

#follow code above to get training predictions, repredicted i know its problematic, classes 2&3 most often misclassified
library(caret)
pred_classes = sapply(1:nrow(trainpreds), function(x) which(trainpreds[x,] == max(trainpreds[x,])))
pred_classes = paste0('Class_',as.character(pred_classes))
confusionMatrix(pred_classes, classes)


#plot overlapping classes 2 & 3 to look for differences
p = cbind(p, classes)
p = as.data.frame(p)
class23 = p[p$classes == 2 | p$classes == 3,]
class23 = class23[,order(colSums(class23), decreasing=T)] #order columns by frequency
class23 = class23[,1:41] #only select features with frequencies >= 10, classes change at id = 16122-16123

library(reshape2)
library(ggplot2)
class23$id = 1:nrow(class23)
class23_m = melt(class23, id = c('id', 'classes'))
ggplot(class23_m, aes(x=variable, y=id, fill=value))+ geom_tile()+geom_hline(aes(yintercept=16122))+
  theme(axis.text.x = element_text(angle = 90, hjust=1))+ylab('Observations - Classes 2 & 3')+
  scale_fill_continuous(high='darkred', low='white', name='Frequency')+xlab('')

class23_m = class23_m[class23_m$variable != 'Num0' & class23_m$variable != 'Num1' & class23_m$variable != 'Num2' & class23_m$variable != 'Num3',]


##svm for classes 2&3 - using all engineered features
library(caret)
train = train[train$class=='Class_2'|train$class=='Class_3',]
idx = createDataPartition(train$class, p=0.40, list=F)
svmTrain = train[idx,]
svmTest = train[-idx,]

svmMod = train(class~., data=svmTrain, method='svmRadial', preProc=c('center','scale'), tuneLength=10, trControl = trainControl(method='repeatedcv', repeats=5))





#visualize in Eigenspace
library(psych)
train = read.csv('engineered1train.csv')
pca.train = scale(train[,-c(1,247)], center=T, scale=T)
pca.train = principal(pca.train, nfactors=10, covar=F)
pca.coords = as.data.frame(pca.train$scores)
pca.coords$class = factor(gsub('Class_','',train$class))

library(rgl)
class234 = pca.coords[pca.coords$class %in% c(2,3,4),]
plot3d(class234[,2], class234[,1], class234[,3], col=class234$class)
legend3d("topright", legend = paste('Class_', c('2', '3', '4')), pch=16, col = unique(class234$class), cex=1, inset=c(0.02))

#see if SVM model separates classes
library(doParallel)
cl = makeCluster(4)
registerDoParallel(cl)

sigmaRangeReduced <- sigest(as.matrix(pca.coords[,-ncol(pca.coords)])) 
svmRGridReduced <- expand.grid(.sigma = sigmaRangeReduced[1], .C = 2^(seq(-4, 4)))

ctrl = trainControl(method='repeatedcv', repeats = 5, classProbs=TRUE, summaryFunction=mcLogLoss)

mcLogLoss <- function (data,
                       lev = NULL,
                       model = NULL) {
  
  if (!all(levels(data[, "pred"]) == levels(data[, "obs"])))
    stop("levels of observed and predicted data do not match")
  
  LogLoss <- function(actual, pred, err=1e-15) {
    pred[pred < err] <- err
    pred[pred > 1 - err] <- 1 - err
    -1/nrow(actual)*(sum(actual*log(pred)))
  }
  
  dtest <- dummyVars(~obs, data=data, levelsOnly=TRUE)
  actualClasses <- predict(dtest, data[,-1])
  
  out <- LogLoss(actualClasses, data[,-c(1:2)])  
  names(out) <- "mcLogLoss"
  out
} 

svmMod = train(pca.coords[,-ncol(pca.coords)], pca.coords$class,
               method = 'svmRadial', metric='mcLogLoss',
               tuneGrid = svmRGridReduced,
               fit = FALSE, trControl=ctrl, maximize=FALSE)

head(pca.coords)

stopCluster(cl)


library(h2o)
localH2O <- h2o.init(nthread=4, Xmx='8g')

train <- read.csv("ottotrain.csv")

#used for blending only
library(caret)
idx = createDataPartition(train$target, p=0.85, list=FALSE)
train = train[idx,]
test = train[-idx,]

for(i in 2:94){
  train[,i] <- as.numeric(train[,i])
  train[,i] <- sqrt(train[,i]+(3/8))
}

#not used for blending
test <- read.csv("ottotest.csv")

for(i in 2:94){
  test[,i] <- as.numeric(test[,i])
  test[,i] <- sqrt(test[,i]+(3/8))
}



train.hex <- as.h2o(localH2O,train)
test.hex <- as.h2o(localH2O,test[,2:94])

predictors <- 2:(ncol(train.hex)-1)
response <- ncol(train.hex)

submission <- read.csv("sampleSubmission.csv")
submission[,2:10] <- 0

for(i in 1:20){
  print(i)
  model <- h2o.deeplearning(x=predictors,
                            y=response,
                            data=train.hex,
                            classification=T,
                            activation="RectifierWithDropout",
                            hidden=c(1024,512,256),
                            hidden_dropout_ratio=c(0.5,0.5,0.5),
                            input_dropout_ratio=0.05,
                            epochs=100,
                            l1=1e-5,
                            l2=1e-5,
                            rho=0.99,
                            epsilon=1e-8,
                            train_samples_per_iteration=2000,
                            max_w2=10,
                            seed=1)
  
  submission[,2:10] <- submission[,2:10] + as.data.frame(h2o.predict(model,test.hex))[,2:10]
  print(i)
  write.csv(submission,file="submission.csv",row.names=FALSE) 
}   
submission = read.csv('submission.csv')

subSums = rowSums(submission[,-1])
submissionNormed = sweep(as.matrix(submission[,-1]), 1, subSums, `/`)
colnames(submissionNormed) = paste0(rep('Class_',9),1:9)
submissionNormed = as.data.frame(submissionNormed)
submissionNormed$obs = test$target
LogLoss(submissionNormed) #0.88

write.csv(train, 'trainforcv.csv')
write.csv(test, 'testforcv.csv')

pred_classes = apply(submissionNormed[,-ncol(submissionNormed)], 1, function(x) which(x == max(x)))
pred_classes = paste0('Class_',as.character(pred_classes))
confusionMatrix(pred_classes, submissionNormed$obs)

#used trainforcv and testforcv. otherwise used script above

param <- list('objective' = 'multi:softprob',
              'eval_metric' = 'mlogloss',
              'num_class' = 9,
              'nthread' = 4,
              'max.depth' = 7)

xgbfinalMod = xgboost(param=param, data = train, label = class, 
                      nfold = 3, nrounds=385, eta=0.1, colsample.bytree=0.5)

#max.depth = 5, nround = 609, eta=0.1, colsample.bytree=1, test ll= 0.490782
#max.depth = 5, nround = 704, eta=0.1, colsample.bytree=0.5, test ll= 0.485484
#max.depth = 5, nround = 871, eta=0.1, colsample.bytree=0.25, test ll= 0.485374


#max.depth = 6, nround = 444, eta=0.1, colsample.bytree=1, test ll= 0.489010
#max.depth = 6, nround = 523, eta=0.1, colsample.bytree=0.5, test ll= 0.482306
#max.depth = 6, nround = 631, eta=0.1, colsample.bytree=0.25, test ll= 0.483093


#max.depth = 7, nround = 341, eta=0.1, colsample.bytree=1, test ll= 0.490034
#max.depth = 7, nround = 385, eta=0.1, colsample.bytree=0.5, test ll= 0.481144** submission8 score = 0.45702 score
#max.depth = 7, nround = 429, eta=0.1, colsample.bytree=0.25, test ll= 0.484539 

xgbfinalMod = xgboost(param=param, data = train, label = class, nrounds=385, eta=0.1, colsample.bytree=0.5)#0.45702 
xgbfinalMod = xgboost(param=param, data = train, label = class, nrounds=463, eta=0.085, colsample.bytree=0.45)#0.45671
xgbfinalMod = xgboost(param=param, data = train, label = class, nrounds=494, eta=0.08, colsample.bytree=0.45)# 0.45637


submit10 = predict(xgbfinalMod,test)
submit10 = matrix(submit10,9,length(submit10)/9)
submit10 = t(submit10)
submit10 = as.data.frame(submit10)
submit10 = cbind(id = 1:nrow(submit10), submit10)
names(submit10) = c('id', paste0('Class_',1:9))
write.csv(submit10, file='submit10.csv', quote=FALSE,row.names=FALSE)

xgbmod.cv = xgb.cv(param=param, data = train, label = class, 
                   nfold = 3, nrounds=1200, eta=0.075, colsample.bytree=0.45)

#max.depth = 7, nround = 379, eta=0.11, colsample.bytree=0.35, test ll= 0.482658
#max.depth = 7, nround = 519, eta=0.11, colsample.bytree=0.15, test ll= 0.489333
#max.depth = 7, nround = 461, eta=0.09, colsample.bytree=0.35, test ll= 0.480999
#max.depth = 7, nround = 611, eta=0.09, colsample.bytree=0.15, test ll= 0.486262
#max.depth = 7, nround = 492, eta=0.085, colsample.bytree=0.35, test ll= 0.480797
#max.depth = 7, nround = 463, eta=0.085, colsample.bytree=0.45, test ll= 0.480152**submission9 - scored 0.45671
#max.depth = 7, nround = 469, eta=0.085, colsample.bytree=0.40, test ll= 0.481215
#max.depth = 7, nround = 544, eta=0.075, colsample.bytree=0.35, test ll= 0.480764
#max.depth = 7, nround = 543, eta=0.08, colsample.bytree=0.35, test ll= 0.482171
#max.depth = 7, nround = 494, eta=0.08, colsample.bytree=0.45, test ll= 0.479735** new best cv score
#max.depth = 7, nround = 533, eta=0.075, colsample.bytree=0.45, test ll= 0.480853



submitblendtune = predict(xgbfinalblendMod,test)
submitblendtune = matrix(submitblendtune,9,length(submitblendtune)/9)
submitblendtune = t(submitblendtune)
submitblendtune = as.data.frame(submitblendtune)
names(submitblendtune) = paste0('Class_',1:9)
write.csv(submitblendtune, file='xgbblendmod.csv', quote=FALSE,row.names=FALSE)

xgbpred_classes = apply(submitblendtune, 1, function(x) which(x == max(x)))
xgbpred_classes = paste0('Class_',as.character(xgbpred_classes))

confusionMatrix(xgbpred_classes, labels)




















####### toy.R
library(nnet)
library(e1071)
library('tree')
library('randomForest')
set.seed(0)
par(mar = c(5,5,2,5))

using.tree <- function(trainX, trainY, ...){
  print("Training Decision tree.")
  trainXY.combined <- cbind(trainX, trainY)
  model <- tree(trainY~., data=trainXY.combined, ...)
}

using.randomForest <- function(trainX, trainY, ...){
  print("Training Forest.")
  trainXY.combined <- cbind(trainX, trainY)
  model <- randomForest(trainY~., data=trainXY.combined, ...)
}

using.svm <- function(trainX, trainY, ...){
  print("Training svm.")
  trainXY.combined <- cbind(trainX, trainY)
  model <- svm(trainY~., data=trainXY.combined, ...)
}

using.nnet <- function(trainX, trainY, ...){
  hiddenLayer_size <- 10
  iterations <- 50
  print("Training Neural Network.")
  print(paste("   Hidden Layer size:", hiddenLayer_size))
  print(paste("   Max iterations:", iterations))
  
  trainY_matrix = class.ind(trainY)
  model <- nnet(x=trainX, y=trainY_matrix, size = hiddenLayer_size, mmaxit = iterations, 
                rang = 0.1, decay = 5e-4, MaxNWts=1000000, linout = FALSE, trace=FALSE, ...)
}

using.bagging <- function(train, trainX, trainY, testX, model.num=5, ...) {
  print(paste("Using Bagging with", model.num, "models."))
  train.num <- dim(trainX)[1]
  if(model.num>1)
    # If number of models is greater than 1, 
    # always take 1/3 more than the average part of training samples.
    train.sub.num <- floor(4./3 * train.num/model.num)
  else
    train.sub.num <- floor(train.num/model.num)
  test.num <- dim(testX)[1]
  result <- data.frame(matrix(0, nrow=test.num, ncol=9))
  for(model.index in 1:model.num){
    train.sub.indices <- sample(1:train.num, train.sub.num)
    train.sub.x <- trainX[train.sub.indices, ]
    train.sub.y <- trainY[train.sub.indices]
    model <- train(train.sub.x, train.sub.y, ...)
    result.tmp = predict(model, testX)
    if(is.factor(result.tmp)){
      result.label <- as.numeric(sub('Class_', '', result.tmp))
    }else{
      result.label <- max.col(result.tmp)
    }
    # for each test sample, accumulate the predicted label vote in the result data frame
    for(test.index in 1:test.num){
      label <- result.label[test.index]
      result[test.index, label] <- result[test.index, label] + 1
    }
  }
  result
}

pca.projecter <- function(pca, original, feature_size){
  projected <- predict(pca, original)
  subtract <- projected[,1:feature_size]
  subtract <- data.frame(subtract)
}

# Computing logloss for a predicted result
compute.logloss <- function(result.df, target.label, test.size, label.size=9){
  logloss <- 0
  # For some classifiers, the predicted results only contains predicted labels, 
  # so we need to construct a matrix just like class.ind() did.
  if(is.factor(result.df)){
    result.matrix <- matrix(0, nrow=test.size, ncol=label.size)
    result.label <- as.numeric(sub('Class_', '', result.df))
    for(sample.index in 1:test.size){
      predictedLabel <- result.label[sample.index]
      result.matrix[sample.index, predictedLabel] <- 1
    }
    result.df <- result.matrix
  } 
  for(sample.index in 1:test.size) {
    outputs <- result.df[sample.index, ]
    # Probability of being classified as one of the labels
    p.i <- outputs/sum(outputs)
    
    targetLabel <- target.label[sample.index]
    # avoid 0 probability
    p.ij <- max(min(p.i[targetLabel], 1-10^-15), 10^-15)
    log_p.ij <- log(p.ij)
    logloss <- logloss + log_p.ij
  }
  logloss <- -logloss/test.size
}

# Computing Balanced Error rate
compute.BERate <- function(result.label, target.label, test.size, label.size){
  ###### Balanced error rate
  ## The index of column means the predicted results,
  ## while the index of row means the real class label.
  BER.matrix <- matrix(0, nrow=label.size, ncol=label.size)
  ## For each sample and predicted results
  for(sample.index in 1:test.size) {
    realLabel <- target.label[sample.index]
    predictedLabel <- result.label[sample.index]
    BER.matrix[predictedLabel, realLabel] = BER.matrix[predictedLabel, realLabel] + 1
  }
  BERate <- 0
  for(label in 1:label.size) {
    Tnum <- BER.matrix[label, label]
    TFnum <- sum(BER.matrix[,label])
    BERate <- BERate + (1-Tnum/TFnum)
  }
  BERate <- BERate/label.size
}

# Prepare plot
Num.rightAxisMax <- 20
plot(seq(1, 100), rep(0, 100), ylim = c(0, Num.rightAxisMax), axes=FALSE, type="n", xlab = NA, ylab = NA)
axis(side=4, at=seq(0, Num.rightAxisMax, by=2))
par(new = T)
plot(seq(1, 100), rep(0, 100), ylim = c(0, 1.0), axes=FALSE, main="Random Forest", 
     type="n", ylab = "", xlab = "Feature Size")
axis(side=1, at=seq(0, 100, by=10))
axis(side=2, at=seq(0, 1, by=0.1))

box()
legend(x=75,y=1,c("Misclass Rate","Balanced Error Rate", "Logloss"),cex=.7, 
       col=c("red","blue","magenta"),pch=c(0,1,2))

# Load data set
inputData <- read.csv('../data/train.csv')
Num.totalSize <- dim(inputData)[1]
Num.totalCol <- dim(inputData)[2]

# Shuffle the row order of the original data
shuffleIndeces <- sample(1:Num.totalSize)
inputData <- inputData[shuffleIndeces, ]

Num.folds <- 3 # K-fold cross-validation
Num.labels <- length(levels(inputData[, Num.totalCol]))
Num.test.size <- floor(Num.totalSize/Num.folds)
fold.size <- floor(Num.totalSize/Num.folds)
Num.train.size <- Num.totalSize-Num.test.size

all.x <- inputData[, 2:(Num.totalCol-1)]
all.y <- inputData[, Num.totalCol]

features.pca <- prcomp(all.x, center = TRUE, scale. = TRUE)
for (features.size in seq(5, 90, 10)){
  print("##############################")
  print(paste("##    PCA feature size", features.size))
  # Gather stats for all the folds
  BERate.list = NULL
  mis_error.list = NULL
  logloss.list = NULL
  for (fold.index in 1:Num.folds) {
    print(paste("---------- Fold #", fold.index, "----------"))
    
    # Create a training and a test set for this fold
    fold.test.indices <- (1+(fold.index-1)*fold.size) : (fold.index * fold.size)
    fold.train.indices <- setdiff(1:Num.totalSize, fold.test.indices)
    fold.test.size <- fold.size
    fold.train.size <- Num.totalSize-fold.test.size
    
    fold.test.x <- all.x[fold.test.indices, ]
    fold.train.x <- all.x[fold.train.indices, ]
    fold.test.y <- all.y[fold.test.indices]
    fold.train.y <- all.y[fold.train.indices]
    
    fold.test.yi <- as.numeric(sub('Class_', '', fold.test.y))
    fold.train.yi <- as.numeric(sub('Class_', '', fold.train.y))
    
    # using PCA to project features to another space
    fold.train.x <- pca.projecter(features.pca, fold.train.x, features.size)
    fold.test.x <- pca.projecter(features.pca, fold.test.x, features.size)
    
    ###### model training and prediction
    ## train can be given as: using.nnet, using.tree, using.randomForest, using.bagging
    train <- using.randomForest
    if(all.equal(train, using.bagging)==TRUE){
      # when calling using.bagging, need to pass the trainer to it as the first argument
      result <- using.bagging(using.tree, fold.train.x, fold.train.y, fold.test.x, model.num=5) 
    }else{
      model <- train(fold.train.x, fold.train.y)
      result <- predict(model, fold.test.x)  
    }
    
    if(is.factor(result)){
      result.label <- as.numeric(sub('Class_', '', result))
    }else{
      result.label <- max.col(result)
    }
    
    ###### mis-classification Error
    mis_error <- mean(as.numeric(fold.test.yi!=result.label))
    mis_error.list <- c(mis_error.list, mis_error)
    print(paste("   Mis-Classification error rate:", mis_error))
    
    ###### Balanced error rate
    BERate <- compute.BERate(result.label, fold.test.yi, fold.test.size, Num.labels)
    BERate.list <- c(BERate.list, BERate)
    print(paste("   Balanced error rate:", BERate))
    
    ###### logloss
    logloss <- compute.logloss(result, fold.test.yi, fold.test.size)
    # Sometimes the predicted output could be all zeros, so computing logloss would 
    # result in devided-by-zero problem, which leads to logloss equals NaN.
    # If this happens, this contagious element is ignored.
    if(!is.na(logloss))
      logloss.list <- c(logloss.list, logloss)  
    print(paste("   Logloss:", logloss))
    
  }
  mis_error.allfolds <- mean(mis_error.list)
  BERate.allfolds <- mean(BERate.list)
  logloss.allfolds <- mean(logloss.list)
  print("===== all folds done, summary below =====")
  print(paste("Mis-Classification error rate: ", mis_error.allfolds))
  print(paste("Balanced error rate: ", BERate.allfolds))
  print(paste("Logloss: ", logloss.allfolds))
  print(paste("##    PCA feature size", features.size))
  print("##############################")
  
  # Plot mis-error rate
  points(features.size, mis_error.allfolds, pch = 0, cex = 0.5, col='red')
  text(features.size, mis_error.allfolds+0.05, round(mis_error.allfolds, 3), cex=0.8, col='red')
  sdev <- sd(mis_error.list)
  arrows(features.size, mis_error.allfolds-sdev, features.size, mis_error.allfolds+sdev, 
         col='red', length=0.05, angle=90, code=3)
  
  # Plot Balanced Error rate
  points(features.size, BERate.allfolds, pch = 1, cex = 0.5, col='blue')
  text(features.size, BERate.allfolds+0.05, round(BERate.allfolds, 3), cex=0.8, col='blue')
  sdev <- sd(mis_error.list)
  arrows(features.size, BERate.allfolds-sdev, features.size, BERate.allfolds+sdev,
         col='blue', length=0.05, angle=90, code=3)
  
  # Plot Logloss rate
  points(features.size, logloss.allfolds/Num.rightAxisMax, pch = 2, cex = 0.5, col='magenta')
  text(features.size, logloss.allfolds/Num.rightAxisMax+0.05, round(logloss.allfolds, 3), cex=0.8, col='magenta')
  sdev <- sd(mis_error.list)
  arrows(features.size, (logloss.allfolds-sdev)/Num.rightAxisMax, features.size, (logloss.allfolds+sdev)/Num.rightAxisMax, 
         col='magenta', length=0.05, angle=90, code=3)
  
}


library(nnet)
library(e1071)
library('tree')
library('lazy')
library('randomForest')
set.seed(0)

using.tree <- function(trainX, trainY){
  print(paste("Using Decision tree."))
  trainXY.combined <- cbind(trainX, trainY)
  model <- tree(trainY~., trainXY.combined)
}

using.randomForest <- function(trainX, trainY){
  print(paste("Using Random Forest."))
  trainXY.combined <- cbind(trainX, trainY)
  model <- randomForest(trainY~., data=trainXY.combined)
}

using.nnet <- function(trainX, trainY){
  hiddenLayer_size <- 10
  iterations <- 50
  print("Using Neural Network.")
  print(paste("Hidden Layer size: ", hiddenLayer_size))
  print(paste("Max iterations: ", iterations))
  
  trainY_matrix = class.ind(trainY)
  model <- nnet(x=trainX, y=trainY_matrix, size = hiddenLayer_size,  mmaxit = iterations, rang = 0.1, decay = 5e-4, MaxNWts=1000000, linout = FALSE, trace=FALSE)
}
# Load data set
inputData <- read.csv('../data/train.csv')
Num.train.totalSize <- dim(inputData)[1]
Num.train.totalCol <- dim(inputData)[2]
X <- inputData[, 2:(Num.train.totalCol-1)]
Y <- inputData[, Num.train.totalCol]
# Load test data set
testData <- read.csv('../data/test.csv')
testData <- testData[,-1]
Num.test.totalSize <- dim(testData)[1]
Num.test.totalCol <- dim(testData)[2]

###### model training and prediction
## train can be given as: using.nnet, using.tree
train <- using.randomForest
model <- train(X, Y)
print(paste("Predicting test data."))
result <- predict(model, testData)
if(is.factor(result)){
  maxindices <- as.numeric(sub('Class_', '', result))
  nameList <- NULL
  for(i in 1:9){
    nameList<-c(nameList, paste("Class_",sep="", i))
  }
  result <- data.frame(matrix(0, nrow=Num.test.totalSize, ncol=9))
  colnames(result)<-nameList
  for(row in 1:Num.test.totalSize){
    result[row, maxindices[row]] <- 1
  }
}

write.csv(result, file = "../data/result.csv")
