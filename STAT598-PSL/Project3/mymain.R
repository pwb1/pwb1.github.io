
#---------------------------------------------------
# Libs
#---------------------------------------------------
library(dplyr)
library(xgboost)


#---------------------------------------------------
# Function to pre-process data
# NOTE: for imputation we do mean(continous vars) and mode(factors)
#---------------------------------------------------
pre_process = function(df)
{
  # #find cols with missing data
  # df.na = data.frame(count = sapply(df,function(x) sum(is.na(x)))) %>% 
  #   mutate(colname = row.names(.)) %>%
  #   filter(count>0)
  
  #function returns mode of factor cols for imputation
  get_mode = function(dfcol)
  {
    df.mode = data.frame(count=cbind(table(dfcol))) %>%
      mutate(colname=row.names(.)) %>%
      filter(count==max(count)) %>%
      select(colname)
    return(df.mode$colname)
  }
  
  #impute factors (mode)
  df$emp_length[is.na(df$emp_length)]=get_mode(df$emp_length)
  df$emp_title[is.na(df$emp_title)]=get_mode(df$emp_title)
  df$title[is.na(df$title)]=get_mode(df$title)
  
  #impute numeric/integer (mean)
  df$dti[is.na(df$dti)]=mean(df$dti, na.rm=TRUE)
  df$revol_util[is.na(df$revol_util)]=mean(df$revol_util, na.rm=TRUE)
  df$mort_acc[is.na(df$mort_acc)]=mean(df$mort_acc, na.rm=TRUE)
  df$pub_rec_bankruptcies[is.na(df$pub_rec_bankruptcies)]=mean(df$pub_rec_bankruptcies, na.rm=TRUE)
  
  #drop columns (e.g. factor level to large, not impactful to model, etc.)
  df = df %>% mutate(emp_title = NULL,
                     title = NULL,
                     zip_code = NULL,
                     earliest_cr_line = NULL)
  
  return(df)
}


#---------------------------------------------------
# Read in train/test and pre-process
# NOTE: train has response var, but test does not, so we treat them
#       slightly different when handling below
#---------------------------------------------------
train = read.csv("train.csv") %>%
  select(loan_status, everything()) %>%
  mutate(loan_status = as.character(loan_status),
         loan_status = ifelse(loan_status=='Fully Paid', 0, 1)) %>%
  pre_process()

test = read.csv("test.csv") %>%
  pre_process()

#convert to matrix + dummy vars for model input
train.x = model.matrix(~.-1, data=train[,-1])
train.y = train[,1]

test.x = model.matrix(~.-1, data=test)
#test.y = 'unknown' to our code - technically will be referenced in grading via label.csv


#---------------------------------------------------
# xgboost model
#---------------------------------------------------
st = Sys.time() #track execution time

#build model
m = xgboost(data = train.x,
            label = train.y,
            nrounds = 100,
            objective = "binary:logistic",
            eval_metric = "logloss",
            verbose = FALSE,
            seed=8544)

#predict and write output to file
p = predict(m, test.x)
p.out = data.frame(cbind(id=test.x[,"id"], prob=p))
write.csv(p.out, "mysubmission1.txt", row.names = FALSE)

#capture runtime
rt.xgboost = difftime(Sys.time(), st, units="min")

#clean-up for next run
rm(m,p,p.out)

