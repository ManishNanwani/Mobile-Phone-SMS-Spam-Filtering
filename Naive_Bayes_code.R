library(tm)
library(SnowballC) #library used for word stemming
library(wordcloud) #used for word visualisation
library(e1071)

sms_data=read.csv("F:\\Aegis\\Machine Learning Foundation\\Machine Learning\\Project\\sms_spam.csv", stringsAsFactors = FALSE)


## Data Visualisation

View(sms_data)
str(sms_data)

## Data Word Cloud for the different levels.
spam= subset(sms_data,type=="spam")
ham=subset(sms_data,type=="ham")

wordcloud(spam$text,min.freq = 15)
wordcloud(ham$text,min.freq = 40)


#Since type is categorical variable, we will convert it into factor
sms_data$type=as.factor(sms_data$type)

str(sms_data)
table(sms_data$type) ##we have 4827 sms's which are ham and remaining 747 as spam

##Data preparation

#Creating corpus of SMS messages
vs=VectorSource(sms_data$text) #Creates the source vector

#Corpus creates volatile corpus-it stores the corpus in memory
#instead of disk and would be destroyed when the R object containing it is destroyed
#as an input we need to give source of the vector
sms_corpus=Corpus(vs)

print(sms_corpus) #we can see that it contains 5574 sms's
inspect(sms_corpus[[1]])
#to view the actual message 


##Data Cleaning

#we will convert entire text in to lower case
sms_corpus_clean = tm_map(sms_corpus, content_transformer(tolower)) 
#as.character(sms_corpus_clean[[1]])

#removing numbers form the msgs
sms_corpus_clean<-tm_map(sms_corpus_clean,removeNumbers)

#removing all the stopwords from msgs
stopwords()
sms_corpus_clean<-tm_map(sms_corpus_clean,removeWords,stopwords())

#removing punctuation
replacePunctuation<-function(x){
  gsub("[[:punct:]]+", " ",x)
}
sms_corpus_clean<-tm_map(sms_corpus_clean,replacePunctuation)

## replacing the words with their root base form
sms_corpus_clean<-tm_map(sms_corpus_clean,stemDocument)

## removing additional whitespace so far generated
sms_corpus_clean<-tm_map(sms_corpus_clean,stripWhitespace)
as.character(sms_corpus_clean[[1]])

## word tokenization 
#The DocumentTermMatrix() function will take a Corpus and create a data structure called a 
#DocumentTermMatrix(DTM) in which rows indicate the Documents and columns indicate terms(words).
sms_dtm=DocumentTermMatrix(sms_corpus_clean)

## Data preparation-splitting into train and test.
sms_dtm_train=sms_dtm[1:3902,]
sms_dtm_test=sms_dtm[3903:5574,]

sms_train_label=sms_data[1:3902,]$type
sms_test_label=sms_data[3903:5574,]$type

## checking proportion of spam and ham in train and test.
prop.table(table(sms_train_label))
prop.table(table(sms_test_label))

##data Visualization-Word Clouds.
wordcloud(sms_corpus_clean,min.freq=50,random.order=FALSE)


##creating indicator features considering only the frequent words.
ncol(sms_dtm)
sms_freq_words=findFreqTerms(sms_dtm_train,5)
length(sms_freq_words)

##filtering the sparse document term matrix to retain just the frquent words.
sms_dtm_freq_train= sms_dtm_train[, sms_freq_words]
sms_dtm_freq_test= sms_dtm_test[, sms_freq_words]

## converting numerical counts to category for the naive bayes classifier 
convert_counts<-function(x){
  x<-ifelse(x>0,"yes",'no')
}

sms_train=apply(sms_dtm_freq_train,MARGIN = 2,convert_counts)
sms_test=apply(sms_dtm_freq_test,MARGIN = 2,convert_counts)


#model buildingg
model=naiveBayes(sms_train,sms_train_label,laplace = 0)
summary(model)

#prediction
p=predict(model,sms_test)
tbl=table("Predicted"=p, "Actual"=sms_test_label)
tbl

library(caret)
confusionMatrix(p,sms_test_label)


## Evaluation Metrics
# Accuracy
accuracy <- sum(diag(tbl)) / sum(tbl)
accuracy

# Precision
precision <- diag(tbl) / rowSums(tbl)
precision

# Recall
recall <- diag(tbl) / colSums(tbl)
recall

# F-1 Score
Fmeasure <- 2 * precision * recall / (precision + recall)
Fmeasure
