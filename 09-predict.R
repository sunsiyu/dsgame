# ===================
# PREDICTION RESULTS
# ===================
# use the random forest model to create a prediction
test <- read.table(testfiles, header=T, sep=",")
id <- test[, 1]
test <- test[,-1]

pred_prob <- predict(lfit[[2]], test, type="prob")
pred_class <- predict(lfit[[2]], test)

submit <- data.frame(id = id, pred_class)
submit$x <- 1
submit <- reshape(submit, idvar = "id", timevar = "pred_class", direction = "wide")
submit <- submit[, c(1, 10, 4, 6, 2, 8, 3, 9, 7, 5)]
submit[is.na(submit)] <- 0
names(submit) <- names(samplesubmission)
#or
submit <- data.frame(id = id, pred_prob)
write.csv(submit, file = "submit-02.csv", row.names = FALSE)
