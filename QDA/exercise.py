import math
import numpy as np
import pandas as pd
import classifierQda as qda
import matplotlib.pyplot as plt

#read data from csv
names = ("Class", "Alcohol","Malic acid","Ash", "Alcalinity of ash" ,"Magnesium","Total phenols","Flavanoids" ,"Nonflavanoid phenols" ,"Proanthocyanins" ,"Color intensity",
"Hue" ,"OD280/OD315 of diluted wines" ,"Proline" )
df = pd.read_csv("wine.csv", names=names)



#split between training and data sets
percentage = 0.7
sample = np.random.rand(len(df)) < percentage
trainSample = df[sample]
testSample = df[~sample]

Class1 = trainSample[trainSample["Class"] == 1]
Class2 = trainSample[trainSample["Class"] == 2]
Class3 = trainSample[trainSample["Class"] == 3]

