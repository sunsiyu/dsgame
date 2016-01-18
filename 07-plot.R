# ========================
# PLOT FEATURE IMPORTANCE
# ========================

preproc_feat_rank <- function(feat_rank, n = nrow(feat_rank))
{
  stopifnot(is.data.frame(feat_rank) && ncol(feat_rank) < 2)
  stopifnot(n > 0)
  
  colnames(feat_rank) <- "attr_importance"
  feat_rank$feature <- rownames(feat_rank)
  rownames(feat_rank) <- 1:nrow(feat_rank)
  feat_rank <- feat_rank[order(-feat_rank$attr_importance), ]
  
  return(feat_rank[1:n, ])
}
  

l_feat_ranks <- list(rank1, rank2, rank3, rank4, rank5, rank7$importance, rank9$importance)
l_feat_ranks <- lapply(l_feat_ranks, preproc_feat_rank, n=20)



# for (i in 1:length(l_feat_ranks)) {
#   p <- ggplot(l_feat_ranks[[i]], aes(reorder(feature, attr_importance), attr_importance)) + 
#     geom_bar(stat="identity", width = 0.1) + coord_flip()
#   ggsave(filename = paste0("feat_rank_", i, ".png"), p)
# }

