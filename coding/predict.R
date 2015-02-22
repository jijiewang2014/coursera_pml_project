# R script to predict the quality of the exercise

library(plyr)
library(dplyr)
library(caret)
library(randomForest)


mydata<-read.csv("data/pml-training.csv", na.strings = "#DIV/0!")

mydata<-mydata[,- grep("kurtosis?|skewness?|max?|min?|amplitude?|var?|avg?|stddev?", names(mydata))  ]

#mydata<-select(mydata,-user_name,-raw_timestamp_part_1,-raw_timestamp_part_2,-cvtd_timestamp,-new_window,-num_window)


inTrain<-createDataPartition(y=mydata$classe,p=0.70,list=FALSE)
training<-mydata[inTrain,]
testing<-mydata[-inTrain,]


preProc<-preProcess(training[,-42], method="pca")
trainPC<-predict(preProc,training[,-42])

modFit<-train(training$classe~.,data=trainPC,method="rpart",prox=TRUE)
modFit
