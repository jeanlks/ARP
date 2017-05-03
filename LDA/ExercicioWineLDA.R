library(MASS)
setwd("~/Documents/Software Engineering/UFG/mestrado/ARP/Aula 3 - LDA,QDA,KNN/LDA")
wine <- read.csv("wine.csv",header=FALSE)
names(wine) <- c("Class", "Alcohol","Malic acid","Ash", "Alcalinity of ash" ,"Magnesium","Total phenols","Flavanoids" ,"Nonflavanoid phenols" ,"Proanthocyanins" ,"Color intensity",
                 "Hue" ,"OD280/OD315 of diluted wines" ,"Proline" )


normalizeBySd <- function(matrix){
  means <- colMeans(matrix)
  for(i in 1:4){
}
#Definicao das funcoes
getValueForEachClass <- function(testVector, testClass, covariance){
  means <- colMeans(testClass)
  vectorResult <- testVector - means;
  inverseCovariance <- solve(covariance)
  firstPart <- vectorResult %*% covariance
  return ( firstPart %*% vectorResult)
}

trainingClass1 <- wine[1:40,]
trainingClass2 <- wine[59:98,]
trainingClass3 <- wine[131:170,]
totalTrainingClass <- do.call(rbind, list(trainingClass1,trainingClass2,trainingClass3) )

testClass1 <- wine[41:49,]
testClass2 <- wine[99:107,]
testClass3 <- wine[171:178,]
totalTestClass <- do.call(rbind, list(testClass1,testClass2,testClass3) )

test <- c(0.060,0.951)
test <- rbind(test, c(-0.357,2.109))
test <- rbind(test,c(0.679,-0.025))
test <- rbind(test,c(0.269,-0.209))


#d1 = getValueForEachClass(testClass1[1,],trainingClass1, cov(wine))

means <- colMeans(testClass1)
vectorResult <- testClass1[1,-1] - means;
inverseCovariance <- solve(cov(wine[,-1]))
firstPart <-  inverseCovariance %*% vectorResult 

