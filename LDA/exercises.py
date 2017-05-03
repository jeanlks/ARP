import numpy as np
import pandas as pd
import LDA
from sklearn import preprocessing

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
#read data from csv
df = pd.read_csv("student-por.csv", sep=";")
columns = ["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob","reason",
                      "guardian", "schoolsup", "famsup", "paid", "activities", "nursery", "higher",
                      "internet", "romantic"]
df = pd.get_dummies(df,columns=columns)
scaled_df = preprocessing.scale(df)

#Formula (XTeste - mediaColuna(X)) * Inversa(MatrizCovariancia) * (XTeste - mediaColuna(X))

#split between training and data sets
percentage = 0.7
sample = np.random.rand(len(scaled_df)) < percentage
train = df[sample]
test = df[~sample]
covarianceMatrix = train.cov()
class1 = train.loc[df['Dalc'] == 1]
class2 = train.loc[df['Dalc'] == 2]
class3 = train.loc[df['Dalc'] == 3]
class4 = train.loc[df['Dalc'] == 4]
class5 = train.loc[df['Dalc'] == 5]

meansClasse1 = class1.mean()
meansClasse2 = class2.mean()
meansClasse3 = class3.mean()
meansClasse4 = class4.mean()
meansClasse5 = class5.mean()
print('valor')
print(test.iloc[10])
print(meansClasse1)
# d1 = LDA.predict(test.iloc[10] ,meansClasse1, covarianceMatrix)
# d2 = LDA.predict(test.iloc[10] ,meansClasse2,  covarianceMatrix)
# d3 = LDA.predict(test.iloc[10] ,meansClasse3, covarianceMatrix)
# d4 = LDA.predict(test.iloc[10] ,meansClasse4, covarianceMatrix)
# d5 = LDA.predict(test.iloc[10] ,meansClasse5, covarianceMatrix)

# print(d1)
# print(d2)
# print(d3)
# print(d4)
# print(d5)





