
#------------------------------------------------------
# Libs
#------------------------------------------------------
st = Sys.time() #track execution time

#code from professor to ensure packages installed in their env when running
mypackages = c("dplyr", "glmnet", "psych", "forcats", "xgboost")   
tmp = setdiff(mypackages, rownames(installed.packages())) 
if (length(tmp) > 0) install.packages(tmp)

#load libs
library(dplyr)
library(glmnet)
library(psych)    #winsorize function
library(forcats)  #combine factor levels
library(xgboost)


#------------------------------------------------------
# Function to winzorize data
# Param: df is dataframe to apply winzorization (on integer columns)
# NOTE: this is seperated from other pre-processing as this step is executed on
#       test/train individually to avoid leakage.
#------------------------------------------------------
winz = function(df)
{
  #Winsorization to handle outliers (NOTE: we start at i=3 to skip Sales_Price and PID)
  for(i in 3:(ncol(df)))
  {
    if(as.character(class(df[,i])=="integer"))
    {
      df[,i] = as.integer(psych::winsor(df[,i], trim = 0.03))
    }
  }
  return(df)
}

#------------------------------------------------------
# Function to Pre-Process data
# Param: df is dataframe to be pre-processed and returned
#------------------------------------------------------
pre_process = function(df)
{
  #move Sales_Price (response var) to front
  df = df %>% select(Sale_Price, everything())
  
  #drop latitude/longitude
  df = df %>% mutate(Longitude=NULL, Latitude=NULL)
  
  #drop garage var to get rid of nulls
  df$Garage_Yr_Blt = NULL
  
  #drop factor columns with dominating values
  df = df %>% mutate(Utilities = NULL,
                     Street = NULL,
                     Pool_QC = NULL,
                     Condition_2 = NULL)
  
  #overall_cond -> combine 9 levels into 3
  df$Overall_Cond = fct_collapse(df$Overall_Cond,
                                 low = c("Very_Poor", "Poor", "Fair"),
                                 med = c("Below_Average","Average", "Above_Average"),
                                 high = c("Good", "Very_Good","Excellent"))
  
  return(df)
}


#------------------------------------------------------
# Read in and pre-process data
#------------------------------------------------------
#read in train/test
train = read.csv('train.csv')
test = read.csv('test.csv')

#combine datasets prior to pre-processing, if we do not it results in different columns in train/test
# due to differing factor levels, check here for detail: https://piazza.com/class/jky28ddlhmu2r8?cid=157
#NOTE: we ONLY do this for pre-processing steps that would not cause leakage
test$Sale_Price = NA #add as a placeholder since Y does not exist in test file; its only in train file
df = rbind(train, test) #combine train/test into single df
df = pre_process(df) #pre-process everything

#split back into seperate train/test and drop the Y placeholder in test
train = df %>% filter(!is.na(Sale_Price))
test = df %>% filter(is.na(Sale_Price)) %>% mutate(Sale_Price=NULL)

#apply winzorization on train/test individually (avoids leakage)
train = winz(train)
test = winz(test)


#------------------------------------------------------
# Lasso Model
#------------------------------------------------------
set.seed(8544)

#train data
train.x = sparse.model.matrix(~.-1, data=train[,-1]) #~.-1 drops intercept created by model.matrix
train.y = log(as.matrix(train[1])) #log transform Y, big IMPACT on accuracy for lasso

#test data
test.x = sparse.model.matrix(~.-1, data=test)

#gen model
cv = cv.glmnet(train.x, train.y, alpha = 1)
m = glmnet(train.x, train.y, alpha = 1, lambda = cv$lambda.min)

#prediction on test
p = exp(predict(m, newx = test.x)) #exp() to reverse log transform that we previously placed on Y

#output to file
output = data.frame(PID = c(test.x[,1]), Sale_Price = c(p))
output$Sale_Price = round(output$Sale_Price, 1) #round to 1 decimal
write.csv(output, "mysubmission1.txt", row.names = FALSE, quote = FALSE)


#------------------------------------------------------
# XGBoost Model
#------------------------------------------------------
set.seed(8544)

#NOTE: we use same train/test data already defined from lasso model in above code

#tuned params for model
params = list(
  eta = 0.05,
  max_depth = 5,
  min_child_weight = 1,
  subsample = 0.65,
  colsample_bytree = .8
)

#build model
model <- xgboost(data = train.x, 
                 label = train.y,
                 params = params,
                 nrounds = 1000, 
                 objective = "reg:linear",
                 verbose = FALSE,
                 seed=8544)

#predict on test  
p = exp(predict(model, test.x))

#output to file
output = data.frame(PID = c(test.x[,1]), Sale_Price = c(p))
output$Sale_Price = round(output$Sale_Price, 1) #round to 1 decimal
write.csv(output, "mysubmission2.txt", row.names = FALSE, quote = FALSE)

#print runtime
paste("runtime:", difftime(Sys.time(), st, units="min"), "minutes")

