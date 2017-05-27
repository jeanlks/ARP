
import pandas as pd



df = pd.read_csv("/Users/Jean/Documents/Software Engineering/UFG/mestrado/ARP/datasets/crimes-in-chicago/Chicago_Crimes_2012_to_2017.csv",sep=",")
#df = pd.read_csv("/Users/Jean/Documents/Software Engineering/UFG/mestrado/ARP/finalProject/datasets/smalldatasetcrimes.csv",sep=",")
df = df[df.Year == 2014]
grouped = df.groupby(['Primary Type'])['Case Number'].count().reset_index(name="count")
grouped = grouped.sort('count',ascending=False)


print(grouped)

