
import pandas as pd
import matplotlib as mpl





df = pd.read_csv("/Users/Jean/Documents/Software Engineering/UFG/mestrado/ARP/datasets/crimes-in-chicago/Chicago_Crimes_2012_to_2017.csv",sep=",")
df = df[df.Year == 2016]

types_to_save = ["THEFT"]

df = df[df['Primary Type'].isin(types_to_save)]


df.boxplot(column="Latitude")

