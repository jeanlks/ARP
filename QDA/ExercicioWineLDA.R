library(MASS)
setwd("~/Documents/Software Engineering/UFG/mestrado/ARP/Aula 3 - LDA,QDA,KNN/LDA")
wine <- read.csv("wine.csv",header=FALSE)
names(wine) <- c("Class", "Alcohol","Malic acid","Ash", "Alcalinity of ash" ,"Magnesium","Total phenols","Flavanoids" ,"Nonflavanoid phenols" ,"Proanthocyanins" ,"Color intensity",
                 "Hue" ,"OD280/OD315 of diluted wines" ,"Proline" )

#Funcoes
#TODO desc
LDA <- function(training, test){
  predictions <- matrix(0,nrow(test),1)
  uniqueClasses <- unique(training[,ncol(training)])
  K = length(uniqueClasses)
  E <- LDAE(training)
  
  U.k <- matrix(0,K,(ncol(training)-1))
  Pi.k <- c(1:K)
  for(k in 1:K){
    U.k[k,] <- Uk(uniqueClasses[k], training)
    Pi.k[k] <- Pik(uniqueClasses[k], training)
  }
  
  #for each test
  for(i in 1:nrow(test)){
    largestValue <- LDADiscriminant(test[i,1:(ncol(test)-1)], E, U.k[1,], Pi.k[1])
    largestValueClass <- uniqueClasses[1]
    #for each class
    for(k in 3:K){
      tempValue <- LDADiscriminant(test[i,1:(ncol(test)-1)], E, U.k[k,], Pi.k[k])
      print(tempValue)
      if(tempValue > largestValue){
        largestValue <- tempValue
        largestValueClass <- uniqueClasses[k]
      }
    }
    #set prediction
    predictions[i] <- largestValueClass
  }
  
  return(predictions)
}

#TODO desc
LDADiscriminant <- function(x, E, U.k, Pi.k){
  return( (unlist(x)%*%(solve(E)))%*%U.k - (1/2)*(U.k%*%(solve(E)))%*%U.k + log(Pi.k) )
}

#TODO desc
LDAE <- function(trainingData){
  numberOfColumns <- ncol(trainingData)
  K = length(unique(trainingData[,numberOfColumns]))
  N = nrow(trainingData)
  
  totalSum = 0
  for(i in 1:N){
    totalSum = totalSum +( unlist((trainingData[i,1:(numberOfColumns-1)]) - Uk(trainingData[i,numberOfColumns],trainingData))%*%t(unlist((trainingData[i,1:(numberOfColumns-1)]) - Uk(trainingData[i,numberOfColumns],trainingData))) )
  }
  
  return(1/(N-K) * totalSum)
}

#Returns a vector of the group mean for each attibute for each observation which belongs to the class.
Uk <- function(class, trainingData){
  numberOfColumns <- ncol(trainingData)
  #as last column is the class
  groupMeanVector <- matrix(0,1,(numberOfColumns-1))
  for(i in 1:(numberOfColumns-1)){
    groupMeanVector[1,i] = (1/Nk(class,trainingData)) * (sum(trainingData[(trainingData[,numberOfColumns] == class),i]))
  }
  return(groupMeanVector)
}

#Returns the number of observations from the provided dataset of the class provided.
Nk <- function(class, trainingData) {
  return ( sum(trainingData == class) )
}

#Returns the prior of the class provided for the provided dataset.
Pik <- function(class, trainingData) {
  return (Nk(class,trainingData)/nrow(trainingData));
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

test <- t(test) * test
