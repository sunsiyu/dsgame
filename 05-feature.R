source("00-data.R")
# ======================
# FEATURE UTILITIES
# ======================
# logical computation
create_feat_xor <- function(df) {
  stopifnot(is.data.frame(df))
  stopifnot(ncol(df) > 1)
  stopifnot(is.numeric(as.matrix(df)))
  n <- ncol(df) - 1
  for (i in 1:n) {
    for (j in (i+1):(n+1)) {
      eval(parse(text = paste0('df$xor_', i, '_', j, ' <- as.integer(xor(df[, i], df[, j]))')))
    }
  }
  return(df)
}

create_feat_diff <- function(df) {
  stopifnot(is.data.frame(df))
  stopifnot(ncol(df) > 1)
  stopifnot(is.numeric(as.matrix(df)))
  n <- ncol(df) - 1
  for (i in 1:n) {
    for (j in (i+1):(n+1)) {
      eval(parse(text = paste0('df$diff_', i, '_', j, ' <- df[, i] - df[, j]')))
    }
  }
  return(df)
}

get_feat_ranks <- function(df, cutoff = 0.7, freqCut = 80/20) {
  cat("calculate correlation matrix ...", fill = T)
  correlationMatrix <- cor(df)
  high_corr <- findCorrelation(correlationMatrix, cutoff = cutoff)
  cat("calculate nzb ...", fill = T)
  nzv <- nzv(df, freqCut = freqCut, saveMetrics = T)
  return(list(high_corr, nzv))
}

# ======================
# FEATURE CONSTRUCTION (OPERATION BETWEEN TWO FEATRUES)
# ======================
# xor use all 93 features
train_feat_xor <- create_feat_xor(trainset[, -94])  # 5sec

feat_ranks_xor <- get_feat_ranks(train_feat_xor)

xnzv <- feat_ranks_xor[[2]]
xnzv <- xnzv[94:nrow(xnzv), ]
xnzv <- xnzv[xnzv$freqRatio > 5, ]
names_out1 <- rownames(xnzv) # 414

xhighcor <- feat_ranks_xor[[1]]
xhighcor <- xhighcor[xhighcor > 93]
names_out2 <- colnames(train_feat_xor)[feat_ranks_xor[[1]]] # 1081

names_out_xor <- unique(c(names_out1, names_out2)) # 1373

train_feat_xor <- train_feat_xor[, -which(colnames(train_feat_xor) %in% names_out_xor)] # 2998 left
saveRDS(train_feat_xor, "train_feat_xor.rds")


# ====================================
# FEATURE CONSTRUCTION (AGGREGATION)
# ====================================
trainset_label <- trainset[, 94]
trainset <- trainset[, -94]
trainset$feat_nzero <- rowSums(trainset == 0)
trainset$feat_rmean <- rowMeans(trainset[, 1:93])
trainset$feat_rmax <- apply(trainset[, 1:93], 1, max)
trainset$feat_rsd <- apply(trainset[, 1:93], 1, sd)
trainset$feat_rdist <- apply(trainset[, 1:93], 1, function(x) sqrt(var(x)*(length(x)-1)))
trainset$feat_rratio <- apply(trainset[, 1:93], 1, function(x) max(x)/min(x[x>0]))
# add label
trainset$target <- trainset_label
trainset_label <- as.factor(as.vector(trainset$target))
cat(c("Trainset with new features: ", dim(trainset), "..."), fill = T)



# ======================
# FEATURE SELECTION
# ======================


correlationMatrix <- cor(trainset)
correlationMatrix

high_corr <- findCorrelation(correlationMatrix, cutoff = 0.7)
high_corr <- high_corr[high_corr > 100]
high_corr  # integer(0)

cat(c(names(trainset)[high_corr], ": were removed ..."), fill = T)
trainset <- trainset[, -high_corr]
cat(c("Trainset after removed high_corr>0.7 features: ", dim(trainset), "..."), fill = T)

nzv <- nzv(trainsetorg[, 1:93], saveMetrics = T)



# =====================================
# FEATURE RANK & SUBSET with FSelector
# =====================================
# chi square test independence between each feature and class
train_feat_xor$target <- trainset$target
# use a subset to run the following - too computational expensivie
intrain_feat_xor <- createDataPartition(train_feat_xor$target, p = 0.1, list = F)
strain_feat_xor <- train_feat_xor[intrain_feat_xor, ]


rank1_xor <- chi.squared(target ~ ., data = strain_feat_xor)
# linear.correlation(target ~ ., data = trainset)  # numeric
# rank.correlation(target ~ ., data = trainset)  # numeric
rank2_xor <- information.gain(target ~ ., data = strain_feat_xor)
rank3_xor <- gain.ratio(target ~ ., data = strain_feat_xor)
rank4_xor <- symmetrical.uncertainty(target ~ ., data = strain_feat_xor)
rank5_xor <- oneR(target ~ ., data = strain_feat_xor)
rank6_xor <- cfs(target ~ ., data = strain_feat_xor)  # subset of feature names
# consistency(target ~ ., data = trainset)  # toolong
# best.first.search(attributes, eval.fun)
# exhaustive.search(attributes, eval.fun)
# forward.search(attributes, eval.fun)
# backward.search(attributes, eval.fun)
# hill.climbing.search(attributes, eval.fun)

# Backwards Feature Selection
# ctrl_rfe <- rfeControl(functions = rfFuncs, method = "cv", number = 10)
# results_rfe <- rfe(trainset[, -ncol(trainset)], 
#                    trainset_label, 
#                    sizes = c(50:60), 
#                    rfeControl = ctrl_rfe)
# results_rfe
# predictors(results)  # return the feature names
# plot(results, type = c("g", "o"))

# =====================================
# FEATURE IMPORTANCE WITH RANDOM FOREST
# =====================================
set.seed(100)
small_trainset <- createDataPartition(trainset$target, p = 0.1, list=F)
small_trainset <- trainset[small_trainset, ]
cat(c("small training set dimension: ", dim(small_trainset), "..."), fill = T)


ctrl_rf <- trainControl(method = 'cv', 
                        number = 5, 
                        verboseIter = T, 
                        classProbs = T, 
                        summaryFunction = LogLossSummary)

grid_rf <- expand.grid(mtry = c(6, 9, 12))
ptm1 <- proc.time()
model_rf_feat <- train(target ~ ., data = small_trainset, 
                       method = 'rf', 
                       metric = 'LogLoss', 
                       maximize = F,
                       tuneGrid = grid_rf, 
                       trControl = ctrl_rf, 
                       ntree = 500)
model_rf_feat$time <- proc.time() - ptm1


model_rf_feat2 <- randomForest(target ~ ., 
                               data = small_trainset, 
                               importance=TRUE, 
                               ntree=100, 
                               mtry=8)

rank7 <- varImp(model_rf_feat)
rank8 <- varImp(model_rf_feat2)



## just to try for the night
ctrl_rf2 <- trainControl(method = 'cv', 
                        number = 5, 
                        verboseIter = T, 
                        classProbs = T, 
                        summaryFunction = LogLossSummary)

grid_rf2 <- expand.grid(mtry = c(5:20))
ptm2 <- proc.time()
model_rf_feat3 <- train(target ~ ., data = trainset, 
                       method = 'rf', 
                       metric = 'LogLoss', 
                       maximize = F,
                       tuneGrid = grid_rf2, 
                       trControl = ctrl_rf2, 
                       ntree = 500)
model_rf_feat3$time <- proc.time() - ptm2
rank9 <- varImp(model_rf_feat3)

source("07-plot.R")
x <- do.call(rbind, l_feat_ranks)
trainset_59_feat <- trainset[, c(colnames(trainset) %in% unique(c(x$feature, rank6, "target")))]

