source("00-data.R")

## logloss
x <- seq(0.0001, 1, by= 0.0001)
vdf <- data.frame(x, log(x))
p <- ggplot(vdf, aes(x = x, y = log.x.)) + 
  geom_line(color = "black", size = 1.5) + 
  theme(axis.title=element_text(size=15), 
        title = element_text(size = 20)) + 
  ggtitle("LogLoss")

## data split
nrow <- c(nrow(trainset), nrow(testset))
vdf <- data.frame(split = as.factor(c("trainset", "testset")), nrow = nrow)
p <- ggplot(vdf, aes(x = split, y = nrow)) + 
  geom_bar(stat = "identity", width = 0.1, fill = "black") + 
  coord_flip() + 
  theme(axis.text = element_text(size = 20),
        axis.title = element_text(size = 20), 
        title = element_text(size = 20)) + 
  ggtitle("Trainset vs Testset")

## one feature
p <- ggplot(trainset, aes(x = feat_92)) + 
  geom_histogram(binwidth = 1) +  
  theme(axis.text = element_text(size = 20),
        axis.title = element_text(size = 20), 
        title = element_text(size = 20)) + 
  ggtitle("Histogram of Feature 92")
p

library(AppliedPredictiveModeling)
transparentTheme(trans = .4)
library(caret)
featurePlot(x = trainset[, 1:4],
            y = trainset$target,
            plot = "pairs",
            ## Add a key at the top
            auto.key = list(columns = 3))



p <- ggplot(tbl_tab, aes(day, walking_speed))
p + geom_line(aes(color = identifier), size = 1) + 
  geom_smooth(color="gray45", size = 1, linetype="dashed") + 
  facet_grid(identifier ~ .) + 
  theme(legend.position="none") + 
  coord_cartesian(ylim=c(0.7, 1.5)) + 
  scale_y_continuous(breaks=seq(0.8, 1.4, 0.3)) + 
  xlab("Day") + 
  ylab("Walking Speed [m/s]") + 
  ggtitle("Time Series Plot of Walking Speed")


### class imbalnace

# distribution of classes
nclasses <- table(train$target)
max(nclasses) / min(nclasses)  # 8.357698
proptbl <- prop.table(table(train$target))
proptbl <- as.data.frame(proptbl)
colnames(proptbl) <- c("class", "frequency")

# ggplot proptbl
p1 <- ggplot(proptbl, aes(reorder(class, frequency), frequency)) + 
  geom_bar(stat="identity", width = 0.1, fill = "skyblue") + 
  coord_flip() +   
  theme(axis.text = element_text(size = 10),
        axis.title = element_text(size = 10), 
        title = element_text(size = 15)) + 
  ggtitle("Class Imbalance")
p1



# ggplot benchmark nn
x <- readRDS("res_model_nn_benchmark.rds")
x <- readRDS("res-nn-benchmark-scale-pca.rds")
names(x)[4] <- "LogLoss_SD"
x1 <- melt(x, id=c("size", "decay"))  # convert to long format
p1 <- ggplot(x, aes(decay)) + 
  geom_line(aes(y = LogLoss), size = 1) + 
  geom_line(aes(y = LogLoss_SD), color = "red", size = 0.9)+ 
  theme(axis.text = element_text(size = 15),
        axis.title = element_text(size = 15), 
        title = element_text(size = 15)) + 
  ggtitle("NN - Parameter Tuning (size=5, proprocess: scale, pca)")
p1

p1 <- ggplot(x, aes(mtry)) + 
  # geom_line(aes(y = LogLoss), size = 1) + 
  geom_line(aes(y = LogLoss_SD), color = "red", size = 0.9)+ 
  theme(axis.text = element_text(size = 15),
        axis.title = element_text(size = 15), 
        title = element_text(size = 15)) + 
  ggtitle("RF - Parameter Tuning Benchmark")
p1
