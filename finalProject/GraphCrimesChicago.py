
import pandas as pd
import numpy as np

import plotly.plotly as py
from plotly.graph_objs import *


colors = {                    'THEFT':'#FF3C33',
                              'BATTERY':'#FF8633',
                              'CRIMINAL DAMAGE':'#B5FF33',
                              'NARCOTICS':'#33FF7A',
                              'ASSAULT':'#33FFC1',
                              'OTHER OFFENSE':'#33FFF0',
                              'BURGLARY':'#33D7FF',
                              'DECEPTIVE PRACTICE':'#33B2FF',
                              'MOTOR VEHICLE THEFT':'#338AFF',
                              'ROBBERY':'#3371FF',
                              'CRIMINAL TRESPASS':'#A233FF',
                              'WEAPONS VIOLATION':'#F333FF',
                              'PUBLIC PEACE VIOLATION':'#524F50',
                              'OFFENSE INVOLVING CHILDREN':'rgb(3, 5, 2)',
                              'PROSTITUTION':'rgb(91, 5, 2)',
                              'CRIM SEXUAL ASSAULT':'rgb(69, 59, 155)',
                              'INTERFERENCE WITH PUBLIC OFFICER':'rgb(217, 224, 155)',
                              'SEX OFFENSE':'rgb(100,100,100)',
                              'HOMICIDE':'rgb(48, 100, 100)',
                              'ARSON':'rgb(50,50,50)',
                              'GAMBLING':'rgb(150,150,150)',
                              'LIQUOR LAW VIOLATION':'rgb(71, 200, 200)',
                              'KIDNAPPING':'rgb(100, 200, 242)',
                              'STALKING':'rgb(205, 132, 96)',
                              'INTIMIDATION':'rgb(50,100,0)',
                              'OBSCENITY':'rgb(200,50,255)',
                              'NON-CRIMINAL':'rgb(200,200,200)',
                              'CONCEALED CARRY LICENSE VIOLATION':'rgb(250,50,75)',
                              'PUBLIC INDECENCY':'rgb(75,25,50)',
                              'NON - CRIMINAL':'rgb(0,0,100)',
                              'OTHER NARCOTIC VIOLATION':'rgb(0,100,125)',
                              'HUMAN TRAFFICKING':'rgb(0,255,75)',
                              'NON-CRIMINAL (SUBJECT SPECIFIED)':'rgb(200,50,20)'}

def getColor(vectorTypesOfCrime):
    idx = 0
    global colors
    colorReturn = []
    for type in vectorTypesOfCrime:
        i = 0
        for key, value in colors.items():
            if(key == type):
                colorReturn.append(value)
                i = i+1
    return colorReturn


df = pd.read_csv("/Users/Jean/Documents/Software Engineering/UFG/mestrado/ARP/datasets/crimes-in-chicago/Chicago_Crimes_2012_to_2017.csv",sep=",")
#df = pd.read_csv("/Users/Jean/Documents/Software Engineering/UFG/mestrado/ARP/finalProject/datasets/smalldatasetcrimes.csv",sep=",")

grouped = df.groupby(['Primary Type','Latitude','Longitude'])['Case Number'].count().reset_index(name="count")
grouped = grouped.sort('count',ascending=False)
grouped.loc[:, 'color'] = pd.Series(getColor(grouped['Primary Type']), index=grouped.index)

print(grouped)




mapbox_access_token = 'pk.eyJ1IjoiamVhbmxrcyIsImEiOiJjaXo1dThlbWswM3VwMndtbmhyNTlyazc3In0.j1ezMv-foA4UUPnJz8DYEA'
py.sign_in('jeanlks','360cSGj1UBUxHAfiDd3M')
data = Data([
    Scattermapbox(
        lat=grouped["Latitude"].iloc[0:40000],
        lon=grouped["Longitude"].iloc[0:40000],
        mode='markers',
        marker=Marker(
            size=grouped['count']/25,
            color = grouped['color']
        ),
        text=grouped['Primary Type'].iloc[0:40000],
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
py.iplot(fig, filename='Primary Type',image='png')
