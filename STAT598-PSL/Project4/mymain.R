
#---------------------------------------------------
# Code required from project instructions
#---------------------------------------------------
all = read.table("data.tsv",stringsAsFactors = F,header = T)
splits = read.table("splits.csv", header = T)
s = 3


#---------------------------------------------------
# Libs
#---------------------------------------------------
#code from professor to ensure packages installed in their env when running
mypackages = c("dplyr", "text2vec", "glmnet")   
tmp = setdiff(mypackages, rownames(installed.packages())) 
if (length(tmp) > 0) install.packages(tmp)

library(dplyr)
library(text2vec)
library(glmnet)

script_rt = Sys.time()
set.seed(8544)


#---------------------------------------------------
# Read in data and split into train/test
#---------------------------------------------------
#0 = bad review, 1 = good review
df = all
df$review = gsub('<.*?>', ' ', df$review)

#get split n
test = df %>% filter(new_id %in% splits[,s])
train = df %>% filter(!(new_id %in% splits[,s]))


#----------------------------
# vocabulary
# NOTE: vocab was built using a 'screening' method provided by professor in piazza post:
#       https://piazza.com/class/jky28ddlhmu2r8?cid=663
# The vocab generated was saved to file and is used in all splits
#----------------------------
vocab_file = read.csv("vocab.txt", stringsAsFactors = FALSE)
terms = vocab_file %>% select(term)
terms = terms$term


#----------------------------
# create train DTM
#----------------------------
it_train = itoken(train$review,
                  preprocessor = tolower,
                  tokenizer = word_tokenizer,
                  ids = train$new_id,
                  progressbar = FALSE)

vocab = create_vocabulary(it_train, ngram = c(1L,2L))

pruned_vocab = prune_vocabulary(vocab,
                                term_count_min = 10,
                                doc_proportion_max = 0.5,
                                doc_proportion_min = 0.001)

vectorizer = vocab_vectorizer(pruned_vocab)

#train DTM
st = Sys.time()
dtm_train = create_dtm(it_train, vectorizer)
print(difftime(Sys.time(), st, units = 'sec'))

#spot checks
#dim(dtm_train)
#identical(rownames(dtm_train), as.character(train$new_id)) #dtm_train stores new_id as char vs numeric


#----------------------------
# create test DTM
#----------------------------
#tokens
it_test = itoken(test$review, 
                 preprocessor = tolower, 
                 tokenizer = word_tokenizer, 
                 ids = test$new_id, 
                 progressbar = FALSE)

#test DTM (we use same vectorizer previously created)
dtm_test = create_dtm(it_test, vectorizer)


#----------------------------
# Apply tf-idf
# #NOTE: tf-idf actually hurt AUC score, not sure why.. but excluded due to decreased score
#----------------------------
# tfidf = TfIdf$new()
# dtm_train = fit_transform(dtm_train, tfidf)
# dtm_test = transform(dtm_test, tfidf)


#----------------------------
# Model - glmnet lasso classifier
#----------------------------

id = which(colnames(dtm_train) %in% terms)

st = Sys.time()
glmnet_classifier = cv.glmnet(x = dtm_train[,id], y = train$sentiment, 
                              family = 'binomial', 
                              alpha = 0, #ridge
                              type.measure = "auc",
                              nfolds = 5,
                              thresh = 1, #1e-3,  # high value is less accurate, but has faster training
                              maxit = .1) #1e3)   # again lower number of iterations for faster training
print(difftime(Sys.time(), st, units = 'sec'))

#check plot/AUC of cv results
#plot(glmnet_classifier)
#print(paste("max AUC =", round(max(glmnet_classifier$cvm), 4)))


#----------------------------
# predict on test
#----------------------------
#predict and check test AUC
preds = predict(glmnet_classifier, newx=dtm_test[,id], s="lambda.min", type = 'response')
glmnet:::auc(test$sentiment, preds)


#----------------------------
# output to file
#----------------------------
output = data.frame(new_id = c(test$new_id), prob = c(preds))
write.csv(output, "mysubmission.txt", row.names = FALSE, quote = FALSE)

print(difftime(Sys.time(), script_rt, units = 'sec'))