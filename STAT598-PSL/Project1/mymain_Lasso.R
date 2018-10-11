
#----------------------------------
# Libs
#----------------------------------
#code from professor to ensure packages installed in their env when running
mypackages = c("dplyr", "glmnet", "psych", "forcats", "xgboost")   
tmp = setdiff(mypackages, rownames(installed.packages())) 
if (length(tmp) > 0) install.packages(tmp)

library(dplyr)
library(glmnet)
library(psych)
library(forcats)

#------------------------------------------------------
# Functions for Lasso via Coordinate Descent
#------------------------------------------------------
one_step_lasso = function(r, x, lam)
{
  xx = sum(x^2)
  xr = sum(r*x)
  b = (abs(xr) -lam/2)/xx
  b = sign(xr)*ifelse(b>0, b, 0)
  return(b)
}

mylasso = function(X, y, lam, n.iter = 100)
{
  p = ncol(X)
  b = data.matrix(rep(0, p))
  row.names(b) = colnames(X)
  r = y
  
  for(step in 1:n.iter)
  {
    for(j in 1:p)
    {
      #update residuals
      r = r + X[, j] * b[j]
      
      #apply 1-step lasso
      b[j] = one_step_lasso(r, X[, j], lam)
      
      #update residuals
      r = r - X[, j] * b[j]
    }
  }
  
  #calculate intercept
  b0 = mean(y)-sum(b*colMeans(X))
  
  #return intercept and betas
  return(c(b0, b))
}

#------------------------------------------------------
# Function to winzorize data
# Param: df is dataframe to apply winzorization (on integer columns)
# NOTE: this is seperated from other pre-processing as this step is executed on
#       test/train individually to avoid leakage.
# Param: df is dataframe to be pre-processed and returned
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
  
  #drop unnecessary columns
  df = df %>% mutate(Utilities = NULL,
                     Street = NULL,
                     Pool_QC = NULL,
                     Condition_2 = NULL,
                     Land_Slope = NULL,
                     Roof_Matl = NULL, 
                     Heating = NULL, 
                     Misc_Feature = NULL, 
                     Low_Qual_Fin_SF = NULL, 
                     Three_season_porch = NULL, 
                     Pool_Area = NULL, 
                     Misc_Val = NULL, 
                     Longitude = NULL,
                     Latitude = NULL,
                     Garage_Yr_Blt = NULL)
  
  #overall_cond -> combine 9 levels into 3
  df$Overall_Cond = fct_collapse(df$Overall_Cond,
                                 low = c("Very_Poor", "Poor", "Fair"),
                                 med = c("Below_Average","Average", "Above_Average"),
                                 high = c("Good", "Very_Good","Excellent"))
  
  return(df)
}


#------------------------------------------------------
# Read in data
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

#prep train
train.y = log(train[,1])
train.x = train[,-1]
train.x = model.matrix(~.-1, data=train.x) #~.-1 to drop intercept created by model.matrix

#prep test
test.x = test #we set test.x = test, as they are now equivelant because Y value is not included in test.csv
test.x = model.matrix(~.-1, data=test.x) #~.-1 to drop intercept created by model.matrix

#------------------------------------------------------
# Build Model
#------------------------------------------------------
#standardize (center/scale features)
#add padding (tiny number) to account for 'Inf' if all vals equivelant (e.g. 0/1)
train.x = scale(train.x + round(runif(nrow(train.x), min = 0, max = .01),3))
test.x = scale(test.x + round(runif(nrow(test.x), min = 0, max = .01),3))

#get Lasso coefficients
m_coef = mylasso(train.x, train.y, lam=exp(2))

#run test data through Lasso model
#we add a default intercept = 1, this is only to keep true intercept in m_coef
# from changing in the matrix multiplication between test.x%*%m_coef when predicting y_hat
test.x = cbind(intercept=1, test.x)

#predict on test
p = exp(test.x %*% m_coef)

#output to file
output = data.frame(PID = c(test[,1]), Sale_Price = c(p)) #we refer test vs test.x here to get PID w/out padding
output$Sale_Price = round(output$Sale_Price, 1) #round to 1 decimal
write.csv(output, "mysubmission3.txt", row.names = FALSE, quote = FALSE)


