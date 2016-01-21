# make sure trainset is the original trainset
source("01-feat-aggregation.R")
# =============================================
# FEATURE CONSTRUCTION (COMBINATION BETWEEN)
# =============================================
# number of important features (by random forest)
nofeat <- c(9, 11, 14, 15, 17, 25, 26, 27, 29, 30, 34, 36, 39, 40, 42, 50, 59, 
            60, 62, 67, 68, 69, 79, 90)
length(nofeat) # 24

# logical computation
create_feat_xor <- function(df, n = 1:ncol(df)) {
  stopifnot(is.numeric(n) && is.vector(n))
  df <- df[, c(n)]
  stopifnot(is.data.frame(df) && (ncol(df)>1) && is.numeric(as.matrix(df)))
  
  n <- ncol(df) - 1
  for (i in 1:n) {
    for (j in (i+1):(n+1)) {
      eval(parse(text = paste0('df$xor_', i, '_', j, ' <- as.integer(xor(df[, i], df[, j]))')))
    }
  }
  df <- df[, grep("xor", names(df))]
  return(df)
}

# difference computation
create_feat_diff <- function(df, n=1:ncol(df)) {
  stopifnot(is.numeric(n) && is.vector(n))
  df <- df[, n]
  stopifnot(is.data.frame(df) && ncol(df)>1 && is.numeric(as.matrix(df)))
  
  n <- ncol(df) - 1
  for (i in 1:n) {
    for (j in (i+1):(n+1)) {
      eval(parse(text = paste0('df$diff_', i, '_', j, ' <- df[, i] - df[, j]')))
    }
  }
  df <- df[, grep("diff", names(df))]
  return(df)
}


df_feat_xor <- create_feat_xor(trainset, n = nofeat)  # 49507x276
df_feat_diff <- create_feat_diff(trainset, n = nofeat)  # 49507x276
cat(c(dim(df_feat_xor), "features with xor were created ..."), fill=T)
cat(c(dim(df_feat_diff), "features with diff were created ..."), fill=T)

# ========================================================================================
# FEATURE RANK (When there are too many, do need to get a feeling of the quality of them) 
# caret, FSelector
# ========================================================================================
# without label column
trainset_all_feats <- cbind(trainset[, -ncol(trainset)], df_feat_xor, df_feat_diff)
dim(trainset_all_feats)  # 49507   651

# RANK 1 - CORRELATION
cor_all <- cor(trainset_all_feats)  # matrix 651 x 651
high_corr <- findCorrelation(cor_all, cutoff = 0.7)

high_corr <- high_corr[high_corr > 99]
length(high_corr)  # 236
# saveRDS(high_corr, "high_corr.rds")
trainset_all_feats <- trainset_all_feats[, -high_corr]  # 49507 x 415

# RANK 2 - nearZeroVar
nzv_all <- nzv(trainset_all_feats, saveMetrics = T)
nzv_all <- as.data.frame(nzv_all)
nzv_all <- nzv_all[94:nrow(nzv_all), ]
nzv_all$feature <- rownames(nzv_all)
rownames(nzv_all) <- 1:nrow(nzv_all)
nzv_all <- nzv_all[order(-nzv_all$freqRatio), ]
head(nzv_all)
nzv_out_names <- nzv_all$feature[1:10]  # 10
# saveRDS(nzv_all, "nzv_all.rds")
trainset_all_feats <- trainset_all_feats[, -which(names(trainset_all_feats) %in% nzv_out_names)]  # 49507 x 405

# RANK 3 - chi square test independence between each feature and class
# add back label
trainset_all_feats$target <- trainset$target
# use a subset to run the following - too computational expensivie
# intrain_feat_rank <- createDataPartition(trainset_all_feats$target, p = 0.1, list = F)
# strainset_all_feats <- trainset_all_feats[intrain_feat_rank, ]
rank3_chi <- chi.squared(target ~ ., data = trainset_all_feats)
rank3_chi_top10 <- cutoff.k(rank3_chi, 10)
saveRDS(rank3_chi, "rank3_chi.rds")
# rank3_chi <- as.data.frame(rank3_chi)
# rank3_chi <- rank3_chi[94:nrow(rank3_chi), ]
# rank3_chi$feature <- rownames(rank3_chi)
# rownames(rank3_chi) <- 1:nrow(rank3_chi)
# rank3_chi <- rank3_chi[order(rank3_chi$attr_importance),]
# rank3_chi_out_names <- rank3_chi$feature[1:20]  # 20
# trainset_all_feats <- trainset_all_feats[, -which(names(trainset_all_feats) %in% rank3_chi_out_names)]


# linear.correlation(target ~ ., data = trainset)  # numeric
# rank.correlation(target ~ ., data = trainset)  # numeric
rank4_ig <- information.gain(target ~ ., data = trainset_all_feats)
rank4_ig_top10 <- cutoff.k(rank4_ig, 10)
saveRDS(rank4_ig, "rank4_ig.rds")
# rank4_ig <- as.data.frame(rank4_ig)
# rank4_ig <- rank4_ig[94:nrow(rank4_ig), ]
# rank4_ig$feature <- rownames(rank4_ig)
# rownames(rank3_chi) <- 1:nrow(rank3_chi)
# rank3_chi <- rank3_chi[order(rank3_chi$attr_importance),]
# rank3_chi_out_names <- rank3_chi$feature[1:20]  # 20
# trainset_all_feats <- trainset_all_feats[, -which(names(trainset_all_feats) %in% rank3_chi_out_names)]


rank5_gain <- gain.ratio(target ~ ., data = trainset_all_feats)
rank5_gain_top10 <- cutoff.k(rank5_gain, 10)
saveRDS(rank5_gain, "rank5_gain.rds")

rank6_sym <- symmetrical.uncertainty(target ~ ., data = trainset_all_feats)
rank6_sym_top10 <- cutoff.k(rank6_sym, 10)
saveRDS(rank6_sym, "rank6_sym.rds")




rank7_oner <- oneR(target ~ ., data = trainset_all_feats)
rank7_oner_top10 <- cutoff.k(rank7_oner, 10)
saveRDS(rank7_oner, "rank7_oner.rds")



rank8_cfs <- cfs(target ~ ., data = strainset_all_feats)  # subset of feature names



# saveRDS(names_xor, "names_xor.rds")

### TOO LONG BELOW METHODS
# consistency(target ~ ., data = trainset)  
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
