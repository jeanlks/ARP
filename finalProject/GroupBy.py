
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("/Users/Jean/Documents/Software Engineering/UFG/mestrado/ARP/datasets/crimes-in-chicago/Chicago_Crimes_2012_to_2017.csv",sep=",")
#df = pd.read_csv("/Users/Jean/Documents/Software Engineering/UFG/mestrado/ARP/finalProject/datasets/smalldatasetcrimes.csv",sep=",")
df = df[df.Year == 2016]

df.Date = pd.to_datetime(df.Date, format='%m/%d/%Y %I:%M:%S %p')


df.loc[:, 'month'] = df['Date'].dt.month
df.loc[:,'day'] = df['Date'].dt.day
df.loc[:,'hour']  = df['Date'].dt.hour
#
# grouped = df.groupby(["Primary Type"])['Case Number'].count().reset_index(name="count")
# grouped = grouped.sort('count',ascending=False)
#
#
# print(grouped)

plt.figure(figsize=(8,10))
df.groupby([df['Primary Type']]).size().sort_values(ascending=True).plot(kind='barh')
plt.title('Number of crimes by type')
plt.ylabel('Crime Type')
plt.xlabel('Number of crimes')
plt.show()