import numpy as np
import pandas as pd
import QDA
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

covClass1 = class1.cov()
covClass2 = class2.cov()
covClass3 = class3.cov()
covClass4 = class4.cov()
covClass5 = class5.cov()
print(train.iloc[:,10].values)
#
# print('valor')
# print(test.as_matrix()[0])
#
# d1 = QDA.predict(test.as_matrix()[0] ,meansClasse1.as_matrix(), covClass1.as_matrix())
# d2 = QDA.predict(test.as_matrix()[0] ,meansClasse2.as_matrix(), covClass2.as_matrix())
# d3 = QDA.predict(test.as_matrix()[0],meansClasse3.as_matrix(),  covClass3.as_matrix())
# d4 = QDA.predict(test.as_matrix()[0],meansClasse4.as_matrix(), covClass4.as_matrix())
# d5 = QDA.predict(test.as_matrix()[0] ,meansClasse5.as_matrix(), covClass5.as_matrix())
#
# print(d1)
# print(d2)
# print(d3)
# print(d4)
# print(d5)
#
#
#
#
#
