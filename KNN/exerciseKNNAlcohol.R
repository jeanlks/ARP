library("mlr")
setwd("~/Documents/Software Engineering/UFG/mestrado/ARP/Aula 3 - LDA,QDA,KNN/knn")
d1=read.table("student-mat.csv",sep=";",header=TRUE)
d2=read.table("student-por.csv",sep=";",header=TRUE)
d3=merge(d1,d2,by=c("school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"))

dadosCategoricos <- c("school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob","reason", 
                      "guardian", "schoolsup", "famsup", "paid", "activities", "nursery", "higher",
                      "internet", "romantic")


for (dadoCategorico in dadosCategoricos) {
 d2 <- createDummyFeatures(d2, cols = dadoCategorico)
}

#Funcoes
euclideanDist <- function(a, b){
  d = 0
  for(i in c(1:(length(a)-1) ))
  {
    d = d + (a[[i]]-b[[i]])^2
  }
  d = sqrt(d)
  return(d)
}


predict <- function(test_data, train_data, k_value){
  
  pred <- c()
  for(i in c(1:nrow(test_data))){
    cat(i,"- ", nrow(test_data))
    distVector = NULL
    classesVector = NULL
    class1 <- 0
    class2 <- 0
    class3 <- 0
    class4 <- 0
    class5 <- 0
    
    for(j in c(1:nrow(train_data))){
      distVector <- c(distVector, euclideanDist(test_data[i,], train_data[j,]))
      classesVector <- c(classesVector, train_data[j,][[10]])
    }
    
    neighborsVector <- data.frame(classesVector, distVector)
    neighborsVector <- neighborsVector[order(neighborsVector$distVector),]
    
    neighborsVector <- neighborsVector[1:k_value,]
    for(k in c(1:k_value)){
      
      if(neighborsVector$classesVector[k] == 1){
        class1 = class1 + 1
      }
      if(neighborsVector$classesVector[k] == 2){
        class2 = class2 + 1
      }
      if(neighborsVector$classesVector[k]==3){
        class3 = class3 +1 ;
      }
      if(neighborsVector$classesVector[k]==4){
        class5 = class5 +1 ;
      }
      if(neighborsVector$classesVector[k]==5){
        class5 = class5 +1 ;
      }
    }
    VectorMaximum <- c(class1,class2,class3,class4,class5)
    
    maximumValue <- which.max(VectorMaximum)
    pred <-  pred <- c(pred, maximumValue)
    print(maximumValue)
    print("---")
  }
  
  return(pred)
  
}

accuracy <- function(classes,results){
  correct = 0
  for(i in c(1:length(results))){
    if(classes[i] == results[i]){
      correct = correct+1
    }
  }
  accu = correct/length(results) * 100
  return(accu)
}


#Split data into training and test sets
smp_size <- floor(0.70 * nrow(d2))
set.seed(123)
train_ind <- sample(seq_len(nrow(d2)), size = smp_size)
train_alcohol <- d2[train_ind, ]
test_alcohol <- d2[-train_ind, ]

#Data prediction
k <- 3

predictions <- predict(test_alcohol,train_alcohol,k)
results <- predictions

#Print accuracy
print(accuracy(test_alcohol[,10],results))
