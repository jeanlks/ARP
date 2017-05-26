require(lattice)
setwd("~/Documents/Software Engineering/UFG/mestrado/ARP/Aula 2 - BAYES/repositorio")
data <- read.csv("iris.csv",header = FALSE);
names(data) <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species")
plot(data)
#Funcoes
gaussiana <- function(x,vetor){
  media = mean(vetor)
  desvioPadrao = sd(vetor)
  exponent <- exp(-(((x-media)^2)/(2*(desvioPadrao^2))))
  return ((1/(sqrt(2*pi)*desvioPadrao))*exponent)
}

bivariate <- function(x,y){

  term1 <- 1 / (2 * pi * sig1 * sig2 * sqrt(1 - rho^2))
  term2 <- (x - mu1)^2 / sig1^2
  term3 <- -(2 * rho * (x - mu1)*(y - mu2))/(sig1 * sig2)
  term4 <- (y - mu2)^2 / sig2^2
  z <- term2 + term3 + term4
  term5 <- term1 * exp((-z / (2 *(1 - rho^2))))
  return (term5)
}


#Funcao de decisao, considera o maior argumento - ARGMAX
decisao <-function(vetorDecisao){
  maior = which.max(vetorDecisao)
  if(maior==1){
    return(paste(c("Classe Setosa, probabilidade : ", vetorDecisao[maior]), collapse = " "))
  }
  if(maior==2){
    return(paste(c("Classe Versicolor, probabilidade :", vetorDecisao[maior]), collapse = " "))
  }
  if(maior==3){
    return(paste(c("Classe Virginica, probabilidade : ", vetorDecisao[maior]), collapse = " "))
  }
}

#Pega a probabilidade por cada classe e realiza o produtorio
getProbabilidadePorClasse <- function(vetorEntrada,vetorDaClasse){
  probabilidade = NULL
  for(i in 1:4){
   probabilidade[i] <- gaussiana(vetorEntrada[i],vetorDaClasse[,i])
  }
  #produtorio
  return(probabilidade[1]*probabilidade[2]*probabilidade[3]*probabilidade[4]*0.33)
}
getProbabilidades <- function(vetorEntrada){
  vetorDeProbabilidades = NULL;
  vetorDeProbabilidades[1] = getProbabilidadePorClasse(vetorEntrada,trainingSetosa[,-5])
  vetorDeProbabilidades[2] = getProbabilidadePorClasse(vetorEntrada,trainingVersicolor[,-5])
  vetorDeProbabilidades[3] = getProbabilidadePorClasse(vetorEntrada,trainingVirginica[,-5])
  return(vetorDeProbabilidades)
}
#Recebe um vetor de entrada e deve retornar as probabilidades de cada elemento
getPrediction <- function(vetorEntrada){
  probabilidades = getProbabilidades(vetorEntrada);
  return(decisao(probabilidades));
}

#Separacao em dados de treinamento e dados de teste
trainingSetosa <- data[1:35,]
trainingVersicolor <- data[51:85,]
trainingVirginica <- data[101:135,]

testSetosa <- data[36:50,]
testVersicolor <- data[86:100,]
testVirginica <- data[136:150,]

classSetosa <-   data[1:50,]
classVersicolor <- data[51:100,]
classVirginica <- data[101:150,]


#Plota Grafico Sepal.Lenght

x<-seq(0, 10, by = 0.1)

a<-gaussiana(x,trainingSetosa$Sepal.Length)
b<-gaussiana(x,trainingVersicolor$Sepal.Length)
c<-gaussiana(x,trainingVirginica$Sepal.Length)

plot(x,a, type="l", lwd=3, ylim=c(0,1.5*max(a,b,c)), xlab="Sepal Length")
lines(x,b, type="l", lwd=3, col="Red")
lines(x,c, type="l", lwd=3, col="Blue")
legend(6, 1.5, legend=c("Setosa", "Versicolor","Virginica"),
       col=c("black", "red","blue"), lwd=3, lty=1:1, cex=0.8)

#Exercicio 1.2 Plotar Graficos
# p(Ci | SL, SW)
# p(Ci | SL, PL)
# p(Ci | SL, PW)
# p(Ci | SW, PL)
# p(Ci | SW, PW)
# p(Ci | PL, PW)

xm <- -3
xp <- 3
ym <- -3
yp <- 3

a <- testSetosa$Petal.Length
b <- testSetosa$Petal.Width

x <- seq(xm, xp, length= as.integer((xp + abs(xm)) * 10))  # vector series x
y <- seq(ym, yp, length= as.integer((yp + abs(ym)) * 10))  # vector series y

sig1 <- var(a)	# variance of x
sig2 <- var(b) #variance of y
rho <- cor(a,b)	# corr(x, y)
mu1 <- median(a)  # expected value of x
mu2 <- median(b)	# expected value of y


z = outer(x,y,bivariate)



#Exercicio2

resultadoSetosaTest = NULL
resultadoVirginicaTest = NULL
resultadoVersicolorTest = NULL
for(i in 1:15){
  resultadoSetosaTest[i] = getPrediction(testSetosa[i,-5])
  resultadoVersicolorTest[i] = getPrediction(testVersicolor[i,-5])
  resultadoVirginicaTest[i] = getPrediction(testVirginica[i,-5])
}

#Probabilidades de teste da classe Setosa
print(resultadoSetosaTest)

#Probabilidades de teste da classe Versicolor
print(resultadoVersicolorTest)

#Probabilidades de teste da classe Virginica
print(resultadoVirginicaTest)

