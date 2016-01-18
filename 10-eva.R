# ===================
# EVALUATION
# ===================
library(pROC)
#code inspired from http://mkseo.pe.kr/stats/?p=790
model_pred <- predict(model, testset, type = "prob")
model_roc <-  roc(testset$class, model_pred$class)

plot(model_roc, 
     print.thres = "best",
     print.thres.best.method="closest.topleft")

model_coords <- coords(model_roc, "best", 
                       best.method="closest.topleft", 
                       ret=c("threshold", "accuracy"))
model_coords