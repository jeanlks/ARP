import math
import numpy as np
import pandas as pd


#read data from csv
df = pd.read_csv("student-por.csv", sep=";")
columns = ["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob","reason",
                      "guardian", "schoolsup", "famsup", "paid", "activities", "nursery", "higher",
                      "internet", "romantic"]
df = pd.get_dummies(df,columns=columns)


#split between training and data sets
percentage = 0.7
sample = np.random.rand(len(df)) < percentage
train = df[sample]
test = df[~sample]

