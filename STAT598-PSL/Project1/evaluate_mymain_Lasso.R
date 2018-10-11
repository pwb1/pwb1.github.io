
#evaluate Lasso model
# Assume the Y value for the test data is stored in a two-column 
# data frame named "test.y":
# col 1: PID
# col 2: Sale_Price
pred <- read.csv("mysubmission3.txt")
test.y = read.csv("Y.csv")
names(test.y)[2] <- "True_Sale_Price"
pred <- merge(pred, test.y, by="PID")
sqrt(mean((log(pred$Sale_Price) - log(pred$True_Sale_Price))^2))


