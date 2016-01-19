# make sure trainset is the original trainset
source("00-data.R")

# =======================================
# FEATURE CONSTRUCTION (ROW AGGREGATION)
# =======================================
label_trainset <- trainset[, 94]
trainset <- trainset[, -94]
trainset$feat_nzero <- rowSums(trainset == 0)
trainset$feat_rmean <- rowMeans(trainset[, 1:93])
trainset$feat_rmax <- apply(trainset[, 1:93], 1, max)
trainset$feat_rsd <- apply(trainset[, 1:93], 1, sd)
trainset$feat_rdist <- apply(trainset[, 1:93], 1, function(x) sqrt(var(x)*(length(x)-1)))
trainset$feat_rratio <- apply(trainset[, 1:93], 1, function(x) max(x)/min(x[x>0]))

# add label
trainset$target <- label_trainset
label_trainset <- as.factor(as.vector(trainset$target))
cat(c("Trainset with new features: ", dim(trainset), "..."), fill = T)


