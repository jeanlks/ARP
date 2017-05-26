
import pandas as pd
import numpy as np

import plotly.plotly as py
from plotly.graph_objs import *






df = pd.read_csv("/Users/Jean/Documents/Software Engineering/UFG/mestrado/ARP/datasets/crimes-in-chicago/Chicago_Crimes_2012_to_2017.csv",sep=",")
#df = pd.read_csv("/Users/Jean/Documents/Software Engineering/UFG/mestrado/ARP/finalProject/datasets/smalldatasetcrimes.csv",sep=",")
df = df[df.Year == 2016]
df = df[df['Primary Type']=='ASSAULT']
grouped = df.groupby(['Latitude','Longitude'])['Case Number'].count().reset_index(name="count")
grouped = grouped.sort('count',ascending=False)

print(grouped)




mapbox_access_token = 'pk.eyJ1IjoiamVhbmxrcyIsImEiOiJjaXo1dThlbWswM3VwMndtbmhyNTlyazc3In0.j1ezMv-foA4UUPnJz8DYEA'
py.sign_in('jeanlks','360cSGj1UBUxHAfiDd3M')
data = Data([
    Scattermapbox(
        lat=grouped["Latitude"],
        lon=grouped["Longitude"],
        mode='markers',
        marker=Marker(
            size=grouped['count']
        ),
        text=grouped['count'],
    )
])
layout = Layout(
    autosize=True,
    hovermode="closest",
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=41.88,
            lon=-87.62
        ),
        pitch=0,
        zoom=10
    ),
)

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='ASSAULT',image='png')
