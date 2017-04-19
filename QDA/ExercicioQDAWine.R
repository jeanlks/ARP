setwd("~/Documents/Software Engineering/UFG/mestrado/ARP/Aula 3 - LDA,QDA,KNN/QDA")
wine <- read.csv("wine.csv",header=FALSE)
names(wine) <- c("Class", "Alcohol","Malic acid","Ash", "Alcalinity of ash" ,"Magnesium","Total phenols","Flavanoids" ,"Nonflavanoid phenols" ,"Proanthocyanins" ,"Color intensity",
                 "Hue" ,"OD280/OD315 of diluted wines" ,"Proline" )

trainingClass1 <- wine[1:40,]
trainingClass2 <- wine[59:98,]
trainingClass3 <- wine[131:170,]
totalTrainingClass <- do.call(rbind, list(trainingClass1,trainingClass2,trainingClass3) )

testClass1 <- wine[41:49,]
testClass2 <- wine[99:107,]
testClass3 <- wine[171:178,]
totalTestClass <- do.call(rbind, list(testClass1,testClass2,testClass3) )

covariance <- cov(trainingClass1[,-1])
