# make sure trainset is the original trainset
source("00-data.R")
source("01-feat-aggregation.R")
# =============================================
# FEATURE CONSTRUCTION (COMBINATION BETWEEN)
# =============================================
# number of important features (by random forest)
nofeat <- c(9, 11, 14, 15, 17, 25, 26, 27, 29, 30, 34, 36, 39, 40, 42, 50, 59, 60, 62, 67, 68, 69, 79, 90)
length(nofeat) # 20

# logical computation
create_feat_xor <- function(df, n = 1:ncol(df)) {
  stopifnot(is.data.frame(df) && ncol(df)>1 && is.numeric(as.matrix(df)))
  stopifnot(is.numeric(n) && is.vector(n))
  df <- df[, n]
  n <- ncol(df) - 1
  tmpdf <- data.frame()
  for (i in 1:n) {
    for (j in (i+1):(n+1)) {
      eval(parse(text = paste0('tmpdf$xor_', i, '_', j, ' <- as.integer(xor(df[, i], df[, j]))')))
    }
  }
  return(tmpdf)
}

# difference computation
create_feat_diff <- function(df, n=1:ncol(df)) {
  stopifnot(is.data.frame(df) && ncol(df)>1 && is.numeric(as.matrix(df)))
  stopifnot(is.numeric(n) && is.vector(n))
  df <- df[, n]
  n <- ncol(df) - 1
  tmpdf <- data.frame()
  for (i in 1:n) {
    for (j in (i+1):(n+1)) {
      eval(parse(text = paste0('tmpdf$diff_', i, '_', j, ' <- df[, i] - df[, j]')))
    }
  }
  return(tmpdf)
}


df_feat_xor <- create_feat_xor(trainset, n = nofeat)  # 190
df_feat_diff <- create_feat_diff(trainset, n = nofeat)  # 190
cat(c(dim(df_feat_xor), "features with xor were created ..."), fill=T)
cat(c(dim(df_feat_diff), "features with diff were created ..."), fill=T)

# ========================================================================================
# FEATURE RANK (When there are too many, do need to get a feeling of the quality of them) 
# caret, FSelector
# ========================================================================================
# without label column
trainset_all_feats <- cbind(trainset[, -ncol(trainset)], df_feat_xor, df_feat_diff)
dim(trainset_all_feats)

# RANK 1 - CORRELATION
cor_all <- cor(trainset_all_feats)
high_corr <- findCorrelation(cor_all, cutoff = 0.7)

high_corr <- high_corr[high_corr > 100]
high_corr  # integer(0)

# RANK 2 - nearZeroVar
nzv_all <- nzv(trainset_all_feats, saveMetrics = T)


trainset <- trainset[, -high_corr]

# RANK 3 - chi square test independence between each feature and class
# add back label
trainset_all_feats$target <- trainset$target
# use a subset to run the following - too computational expensivie
intrain_feat_rank <- createDataPartition(trainset_all_feats$target, p = 0.3, list = F)
strainset_all_feats <- trainset_all_feats[intrain_feat_rank, ]


rank3_chi <- chi.squared(target ~ ., data = strainset_all_feats)
# linear.correlation(target ~ ., data = trainset)  # numeric
# rank.correlation(target ~ ., data = trainset)  # numeric
rank2_xor <- information.gain(target ~ ., data = strain_feat_xor)
rank3_xor <- gain.ratio(target ~ ., data = strain_feat_xor)
rank4_xor <- symmetrical.uncertainty(target ~ ., data = strain_feat_xor)
rank5_xor <- oneR(target ~ ., data = strain_feat_xor)
rank6_xor <- cfs(target ~ ., data = strain_feat_xor)  # subset of feature names

rank2_xor_df2[1:50,]
rank3_xor_df <- as.data.frame(rank3_xor)
rank3_xor_df$feature <- rownames(rank3_xor)
rank3_xor_df <- rank3_xor_df[order(-rank3_xor_df$attr_importance), ]
rank3_xor_df[1:50,]

names_xor <- unique(c(rank2_xor_df2[1:50,"feature"], rank3_xor_df[1:50,"feature"]))
saveRDS(names_xor, "names_xor.rds")

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
