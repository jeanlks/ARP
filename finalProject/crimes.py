import pandas as pd
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import math

dataset  = pd.read_csv("/Users/Jean/Documents/Software Engineering/UFG/mestrado/ARP/finalProject/datasets/Crimes_-_2001_to_present.csv")
columns = dataset.columns.values.tolist()

grouped = dataset.groupby(['Latitude','Longitude'])

print(columns)
latitudes = dataset['Latitude']
longitudes = -dataset['Longitude']

fig, ax = plt.subplots()
earth = Basemap(ax=ax)
earth.drawcoastlines(color='#556655', linewidth=0.5)
ax.scatter(grouped['Longitude'], grouped['Longitude'], grouped['Primary Type'].size() ** 2,
           c='red', alpha=0.5, zorder=10)
ax.set_xlabel("This month's 4.5M+ earthquakes")
fig.savefig('usgs-monthly-4.5M.png')



