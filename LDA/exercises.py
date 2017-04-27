import numpy as np
import pandas as pd
import LDA

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
#read data from csv
df = pd.read_csv("student-por.csv", sep=";")
columns = ["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob","reason",
                      "guardian", "schoolsup", "famsup", "paid", "activities", "nursery", "higher",
                      "internet", "romantic"]
df = pd.get_dummies(df,columns=columns)


#split between training and data sets
percentage = 0.7
samplemean = LDA.normalizationBySd(df.as_matrix())
print(samplemean)
sample = np.random.rand(len(df)) < percentage
train = df[sample]
test = df[~sample]


