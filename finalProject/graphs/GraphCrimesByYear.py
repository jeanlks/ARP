
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("/users/jean/documents/software engineering/ufg/mestrado/arp/datasets/crimes-in-chicago/chicago_crimes_2012_to_2017.csv",sep=",")
grouped = df.groupby(['primary type','year'])['case number']

df.date = pd.to_datetime(df.date, format='%m/%d/%y %i:%m:%s %p')
df.index = pd.datetimeindex(df.date)

print(grouped)

plt.figure(figsize=(11,4))
df.resample('d').size().rolling(365).sum().plot()
plt.title('rolling sum of all crimes from 2005 - 2016')
plt.ylabel('number of crimes')
plt.xlabel('days')
plt.show()